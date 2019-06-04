import os
import sys
import argparse
from random import shuffle

import net
import data_preprocess as pp
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torch.utils.data import DataLoader

from tool.log import LogManager
from tool.loss import nllloss, l1loss,l2loss, kl_for_vae
from tool.kaldi.kaldi_manager import read_feat, read_vec

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default="data/timit/train", type=str)
parser.add_argument("--dev_dir", default="data/timit/dev", type=str)
parser.add_argument("--model_dir", default="model/vae_timit", type=str)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--size", default=1, type=int)
args = parser.parse_args()
#####################################################################
train_dir = args.train_dir
dev_dir = args.dev_dir
model_dir = args.model_dir
#####################################################################
epochs = 1000
batch_size = 2
frame_size = 128
step_size = 64

lr_e = 0.00002
lr_g = 0.00002
lr_d = 0.00001

change_epoch = 2000
save_per_epoch = 5

adv_coef = 1.0; spk_coef = 3.0; cyc_coef = 10.0; con_coef = 1.0; norm_coef=0.01
#####################################################################
torch.cuda.empty_cache()
os.system("mkdir -p "+ model_dir +"/parm")

train_storage = read_feat(train_dir+"/feats.ark", cmvn=True, delta=True)
dev_storage = read_feat(dev_dir+"/feats.ark", cmvn=True, delta=True)

# read ivector into {spk_id:ivector} 
train_ivecs = read_vec(train_dir+"/ivectors.ark")
dev_ivecs = read_vec(dev_dir+"/ivectors.ark")

# {spk_id:[segment(128*120)1, seg2, seg3, ...]}
train_segs = pp.make_spk_cnn_set(train_storage, frame_size=frame_size, step_size=step_size); print(len(train_segs))
dev_segs = pp.make_spk_cnn_set(dev_storage, frame_size=frame_size, step_size=step_size); print(len(dev_segs))
spk_dim = len(train_segs.keys())

train_set = []
for spk_id, seg_set in train_segs.items():
    seg_len=len(seg_set)
    train_set.extend([(spk_id, seg_idx) for seg_idx in range(seg_len)])
dev_set = []
for spk_id, seg_set in dev_segs.items():
    seg_len=len(seg_set)
    dev_set.extend([(spk_id, seg_idx) for seg_idx in range(seg_len)])

# for D spk training
train_spks = list(train_segs.keys())
train_spkidx = {spk_id: idx for idx, spk_id in enumerate(train_spks)}

train_loader = DataLoader(train_set, batch_size=batch_size*2, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=batch_size*2)

VAE_E = net.Generator_CNN(feat_dim=120, aux_dim=100)
VAE_G = net.Generator_CNN(feat_dim=120, aux_dim=100)
VAE_D_tf = net.Discriminator_MLP(feat_dim=120, hidden_dim=512)
VAE_D_spk = net.Discriminator_CNN(feat_dim=120, frame_dim=128, hidden_dim=512, spk_dim=spk_dim)

E_opt = optim.Adam(VAE_E.parameters(), lr=lr_e)
G_opt = optim.Adam(VAE_G.parameters(), lr=lr_g)
Dtf_opt = optim.Adam(VAE_D_tf.parameters(), lr=lr_d)
Dspk_opt = optim.Adam(VAE_D_spk.parameters(), lr=lr_d)

E_sch = scheduler.StepLR(E_opt, step_size=100, gamma=0.5)
G_sch = scheduler.StepLR(G_opt, step_size=100, gamma=0.5)
Dtf_sch = scheduler.StepLR(Dtf_opt, step_size=100, gamma=0.5)
Dspk_sch = scheduler.StepLR(Dspk_opt, step_size=100, gamma=0.5)

for model in [VAE_E, VAE_G, VAE_D_tf, VAE_D_spk]:
    model.cuda()
torch.save(VAE_E, model_dir+"/init.pt")

# Cycle Phase
print("Cycle GAN phase")
lm = LogManager()
for stype in ["D_adv", "D_spk", "G_adv", "G_spk", "cyc", "content", "D_norm", "G_norm"]:
    lm.alloc_stat_type(stype)

for epoch in range(epochs):
    print("EPOCH :", epoch)
    # coefficient change
    if epoch >= change_epoch:
        for sch in [E_sch, G_sch, Dtf_sch, Dspk_sch]:
            sch.step()
    # Train
    lm.init_stat()
    for model in [VAE_E, VAE_G, VAE_D_tf, VAE_D_spk]:
        model.train()
    # D & SPK phase
    for spk_ids, idxs in train_loader:
        total_len = len(spk_ids)
        if total_len % 2 == 0: # even
            start_A = 0; start_B = total_len//2
            end_A = start_B; end_B = total_len
        else: # odd
            start_A = 0; start_B = total_len//2
            end_A = start_B + 1; end_B = total_len

        dA = [(spk_ids[i], idxs[i]) for i in range(start_A, end_A)]
        dB = [(spk_ids[i], idxs[i]) for i in range(start_B, end_B)]
        xA = [train_segs[spk_id][idx] for spk_id, idx in dA]; xA = torch.Tensor(xA).float().cuda() # (batch_size, 128, 120)
        xB = [train_segs[spk_id][idx] for spk_id, idx in dB]; xB = torch.Tensor(xB).float().cuda() # (batch_size, 128, 120)
        iA = [train_ivecs[spk_id] for spk_id, idx in dA]; iA = torch.Tensor(iA).float().cuda() # (batch_size, 100)
        iB = [train_ivecs[spk_id] for spk_id, idx in dB]; iB = torch.Tensor(iB).float().cuda() # (batch_size, 100)
        tA = [train_spkidx[spk_id] for spk_id, idx in dA]; tA = torch.Tensor(tA).long().cuda() # (batch_size)
        tB = [train_spkidx[spk_id] for spk_id, idx in dB]; tB = torch.Tensor(tB).long().cuda() # (batch_size)
        
        cA = VAE_E(xA, iA)
        xA2B = VAE_G(cA, iB)
        cB = VAE_E(xB, iB)
        xB2A = VAE_G(cB, iA)
        
        tf_xA = VAE_D_tf(xA); spk_xA = VAE_D_spk(xA)
        tf_xB = VAE_D_tf(xB); spk_xB = VAE_D_spk(xB)
        tf_xA2B = VAE_D_tf(xA2B); spk_xA2B = VAE_D_spk(xA2B)
        tf_xB2A = VAE_D_tf(xB2A); spk_xB2A = VAE_D_spk(xB2A)
        spk_cA = VAE_D_spk(cA)
        spk_cB = VAE_D_spk(cB)

        D_adv = torch.mean(tf_xA2B) - torch.mean(tf_xA) + torch.mean(tf_xB2A) - torch.mean(tf_xB)
        D_spk = nllloss(spk_xA, tA) + nllloss(spk_xB, tB) + nllloss(spk_xA2B, tA) + nllloss(spk_xB2A, tB)
        D_norm = nllloss(spk_cA, tA) + nllloss(spk_cB, tB)

        # Update Parameter
        total = adv_coef * D_adv + spk_coef * D_spk + norm_coef * D_norm
        for opt in [Dtf_opt, Dspk_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [Dtf_opt, Dspk_opt]:
            opt.step()
        
        # Save to Log
        lm.add_torch_stat("D_adv", D_adv)
        lm.add_torch_stat("D_spk", D_spk)
        lm.add_torch_stat("D_norm", D_norm)
    # G phase
    for spk_ids, idxs in train_loader:
        total_len = len(spk_ids)
        if total_len % 2 == 0: # even
            start_A = 0; start_B = total_len//2
            end_A = start_B; end_B = total_len
        else: # odd
            start_A = 0; start_B = total_len//2
            end_A = start_B + 1; end_B = total_len

        dA = [(spk_ids[i], idxs[i]) for i in range(start_A, end_A)]
        dB = [(spk_ids[i], idxs[i]) for i in range(start_B, end_B)]
        xA = [train_segs[spk_id][idx] for spk_id, idx in dA]; xA = torch.Tensor(xA).float().cuda() # (batch_size, 128, 120)
        xB = [train_segs[spk_id][idx] for spk_id, idx in dB]; xB = torch.Tensor(xB).float().cuda() # (batch_size, 128, 120)
        iA = [train_ivecs[spk_id] for spk_id, idx in dA]; iA = torch.Tensor(iA).float().cuda() # (batch_size, 100)
        iB = [train_ivecs[spk_id] for spk_id, idx in dB]; iB = torch.Tensor(iB).float().cuda() # (batch_size, 100)
        tA = [train_spkidx[spk_id] for spk_id, idx in dA]; tA = torch.Tensor(tA).long().cuda() # (batch_size)
        tB = [train_spkidx[spk_id] for spk_id, idx in dB]; tB = torch.Tensor(tB).long().cuda() # (batch_size)
        
        cA = VAE_E(xA, iA)
        xA2B = VAE_G(cA, iB)
        cB = VAE_E(xB, iB)
        xB2A = VAE_G(cB, iA)
        
        tf_xA = VAE_D_tf(xA); spk_xA = VAE_D_spk(xA)
        tf_xB = VAE_D_tf(xB); spk_xB = VAE_D_spk(xB)
        tf_xA2B = VAE_D_tf(xA2B); spk_xA2B = VAE_D_spk(xA2B)
        tf_xB2A = VAE_D_tf(xB2A); spk_xB2A = VAE_D_spk(xB2A)
        spk_cA = VAE_D_spk(cA)
        spk_cB = VAE_D_spk(cB)

        G_adv = torch.mean(tf_xA) - torch.mean(tf_xA2B) + torch.mean(tf_xB) - torch.mean(tf_xB2A)
        G_spk = nllloss(spk_xA, tA) + nllloss(spk_xB, tB) +nllloss(spk_xA2B, tB) + nllloss(spk_xB2A, tA)
        G_norm = - nllloss(spk_cA, tA) - nllloss(spk_cB, tB)

        # phase 2: A2B -> A2B2A
        cA2B = VAE_E(xA2B, iB) 
        xA2B2A = VAE_G(cA2B, iA)
        cB2A = VAE_E(xB2A, iA)
        xB2A2B = VAE_G(cB2A, iB)

        cyc = l1loss(xA2B2A, xA) + l1loss(xB2A2B, xB)
        con_cyc = l1loss(cA2B, cA) + l1loss(cB2A, cB)
        
        # Update Parameter
        total = adv_coef * G_adv + spk_coef * G_spk + cyc_coef * cyc + con_coef * con_cyc + norm_coef * G_norm
        for opt in [E_opt, G_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [E_opt, G_opt]:
            opt.step()

        # Save to Log
        lm.add_torch_stat("G_adv", G_adv)
        lm.add_torch_stat("G_spk", G_spk)
        lm.add_torch_stat("cyc", cyc)
        lm.add_torch_stat("content", con_cyc)
        lm.add_torch_stat("G_norm", G_norm)

    # Print total log of train phase
    print("Train G > \n\t", end='')
    lm.print_stat()
    # Eval
    lm.init_stat()
    for model in [VAE_E, VAE_G, VAE_D_tf, VAE_D_spk]:
        model.eval()
    with torch.no_grad():
        for spk_ids, idxs in dev_loader:
            total_len = len(spk_ids)
            if total_len % 2 == 0: # even
                start_A = 0; start_B = total_len//2
                end_A = start_B; end_B = total_len
            else: # odd
                start_A = 0; start_B = total_len//2
                end_A = start_B + 1; end_B = total_len

            dA = [(spk_ids[i], idxs[i]) for i in range(start_A, end_A)]
            dB = [(spk_ids[i], idxs[i]) for i in range(start_B, end_B)]
            xA = [dev_segs[spk_id][idx] for spk_id, idx in dA]; xA = torch.Tensor(xA).float().cuda() # (batch_size, 128, 120)
            xB = [dev_segs[spk_id][idx] for spk_id, idx in dB]; xB = torch.Tensor(xB).float().cuda() # (batch_size, 128, 120)
            iA = [dev_ivecs[spk_id] for spk_id, idx in dA]; iA = torch.Tensor(iA).float().cuda() # (batch_size, 100)
            iB = [dev_ivecs[spk_id] for spk_id, idx in dB]; iB = torch.Tensor(iB).float().cuda() # (batch_size, 100)
            
            cA = VAE_E(xA, iA)
            xA2B = VAE_G(cA, iB)
            cB = VAE_E(xB, iB)
            xB2A = VAE_G(cB, iA)
            
            tf_xA = VAE_D_tf(xA)
            tf_xB = VAE_D_tf(xB)
            tf_xA2B = VAE_D_tf(xA2B)
            tf_xB2A = VAE_D_tf(xB2A)
            spk_cA = VAE_D_spk(cA)
            spk_cB = VAE_D_spk(cB)

            adv = torch.mean(tf_xA) - torch.mean(tf_xA2B) + torch.mean(tf_xB) - torch.mean(tf_xB2A)
            norm = nllloss(spk_cA, tA) + nllloss(spk_cB, tB)
            
            cA2B = VAE_E(xA2B, iB) 
            xA2B2A = VAE_G(cA2B, iA)
            cB2A = VAE_E(xB2A, iA)
            xB2A2B = VAE_G(cB2A, iB)

            cyc = l1loss(xA2B2A, xA) + l1loss(xB2A2B, xB)
            con_cyc = l1loss(cA2B, cA) + l1loss(cB2A, cB)

            # Save to Log
            lm.add_torch_stat("G_adv", adv)
            lm.add_torch_stat("G_norm", norm)
            lm.add_torch_stat("cyc", cyc)
            lm.add_torch_stat("content", con_cyc)

    # Print total log of train phase
    print("Eval > \n\t", end='')
    lm.print_stat()
    # Save per defined epoch
    if int(epoch % save_per_epoch) == 0:
        parm_path = model_dir+"/parm/"+str(epoch)+".pt"
        torch.save(VAE_E.state_dict(), parm_path)
