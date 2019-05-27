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

adv_coef = 1.0; spk_coef = 10.0; cyc_coef = 5.0; kl_coef = 1.0
#####################################################################
torch.cuda.empty_cache()
os.system("mkdir -p "+ model_dir +"/parm")

train_storage = read_feat(train_dir+"/feats.ark", delta=True)
dev_storage = read_feat(dev_dir+"/feats.ark", delta=True)

# read ivector into {spk_id:ivector} 
train_ivecs = read_vec(train_dir+"/ivectors.ark")
dev_ivecs = read_vec(dev_dir+"/ivectors.ark")

for dataset in [train_storage, dev_storage]:
    for utt_id, feat_mat in dataset.items():
        feat_mat = pp.matrix_normalize(feat_mat, axis=1, fcn_type="mean")
        dataset[utt_id] = feat_mat

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

VAE_E = net.VAE_Encoder(feat_dim=120)
VAE_S = net.VAE_Speaker(feat_dim=100, hidden_dim=512, output_dim=480)
VAE_G = net.VAE_Generator(feat_dim=120)
VAE_D = net.VAE_Discriminator(feat_dim=120, hidden_dim=512, spk_dim=spk_dim)


E_opt = optim.Adam(VAE_E.parameters(), lr=lr_e)
S_opt = optim.Adam(VAE_S.parameters(), lr=lr_g)
G_opt = optim.Adam(VAE_G.parameters(), lr=lr_g)
D_opt = optim.Adam(VAE_D.parameters(), lr=lr_d)

E_sch = scheduler.StepLR(E_opt, step_size=100, gamma=0.5)
S_sch = scheduler.StepLR(S_opt, step_size=100, gamma=0.5)
G_sch = scheduler.StepLR(G_opt, step_size=100, gamma=0.5)
D_sch = scheduler.StepLR(D_opt, step_size=100, gamma=0.5)

for model in [VAE_E, VAE_S, VAE_G, VAE_D]:
    model.cuda()

lm = LogManager()
for stype in ["D_adv", "D_spk", "G_adv", "G_spk", "cyc", "KL"]:
    lm.alloc_stat_type(stype)

torch.save(VAE_E, model_dir+"/init.pt")

# In train: get (spk_id, idx) * (batch_size * 2)
# idxA = (batch_size, (spk_id, idx)); idxB = (batch_size, (spk_id, idx))
# xA = [train_segs[spk_id, idx] for spk_id, idx in idxA]
# iA = [train_ivecs[spk_id] for spk_id, idx in idxA]
# tA = [train_spkidx[spk_id] for spk_id, idx in idxA]

for epoch in range(epochs):
    print("EPOCH :", epoch)
    # coefficient change
    if epoch >= change_epoch:
        for sch in [E_sch, S_sch, G_sch, D_sch]:
            sch.step()
    # Train
    lm.init_stat()
    for model in [VAE_E, VAE_S, VAE_G, VAE_D]:
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
        
        conA = VAE_E(xA); meanB, stdB, spkB = VAE_S(iB); A2B = VAE_G(conA, spkB)
        conB = VAE_E(xB); meanA, stdA, spkA = VAE_S(iA); B2A = VAE_G(conB, spkA)
        
        ansT1, ansA1 = VAE_D(xA)
        ansT2, ansB1 = VAE_D(xB)
        ansF1, ansA2 = VAE_D(A2B)
        ansF2, ansB2 = VAE_D(B2A)

        D_adv = l2loss(ansT1, 1) / 2 + l2loss(ansT2, 1) / 2 + l2loss(ansF1, 0) / 2 + l2loss(ansF2, 0) / 2 
        D_spk = nllloss(ansA1, tA) + nllloss(ansB1, tB) + nllloss(ansA2, tA) + nllloss(ansB2, tB)

        # Update Parameter
        total = adv_coef * D_adv + spk_coef * D_spk
        D_opt.zero_grad()
        total.backward()
        D_opt.step()

        # Save to Log
        lm.add_torch_stat("D_adv", D_adv)
        lm.add_torch_stat("D_spk", D_spk)
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
        
        # phase 1: xA -> A2B
        conA = VAE_E(xA); meanB, stdB, spkB = VAE_S(iB); A2B = VAE_G(conA, spkB)
        conB = VAE_E(xB); meanA, stdA, spkA = VAE_S(iA); B2A = VAE_G(conB, spkA)
        
        ansT1, ansB1 = VAE_D(A2B)
        ansT2, ansA1 = VAE_D(B2A)

        G_adv = l2loss(ansT1, 1) / 2 + l2loss(ansT2, 1) / 2
        G_spk = nllloss(ansA1, tA) + nllloss(ansB1, tB)

        # phase 2: A2B -> A2B2A
        conA2B = VAE_E(A2B); A2B2A = VAE_G(conA2B, spkA)
        conB2A = VAE_E(B2A); B2A2B = VAE_G(conB2A, spkB)

        cyc = l1loss(A2B2A, xA) + l1loss(B2A2B, xB)
        kl = kl_for_vae(meanA, stdA) + kl_for_vae(meanB, stdB)

        # Update Parameter
        total = adv_coef * G_adv + spk_coef * G_spk + cyc_coef * cyc + kl_coef * kl
        for opt in [E_opt, S_opt, G_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [E_opt, S_opt, G_opt]:
            opt.step()

        # Save to Log
        lm.add_torch_stat("G_adv", G_adv)
        lm.add_torch_stat("G_spk", G_spk)
        lm.add_torch_stat("cyc", cyc)
        lm.add_torch_stat("KL", kl)

    # Print total log of train phase
    print("Train G > \n\t", end='')
    lm.print_stat()
    # Eval
    lm.init_stat()
    for model in [VAE_E, VAE_S, VAE_G, VAE_D]:
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
            
            # phase 1: xA -> A2B
            conA = VAE_E(xA); meanB, stdB, spkB = VAE_S(iB); A2B = VAE_G(conA, spkB)
            conB = VAE_E(xB); meanA, stdA, spkA = VAE_S(iA); B2A = VAE_G(conB, spkA)
            
            ansxA, _ = VAE_D(xA)
            ansxB, _ = VAE_D(xB)
            ansA2B, _ = VAE_D(A2B)
            ansB2A, _ = VAE_D(B2A)

            D_adv = l2loss(ansxA, 1) / 2 + l2loss(ansxB, 1) / 2 + l2loss(ansA2B, 0) / 2 + l2loss(ansB2A, 0) / 2 
            G_adv = l2loss(ansA2B, 1) / 2 + l2loss(ansB2A, 1) / 2
            
            # phase 2: A2B -> A2B2A
            conA2B = VAE_E(A2B); A2B2A = VAE_G(conA2B, spkA)
            conB2A = VAE_E(B2A); B2A2B = VAE_G(conB2A, spkB)

            cyc = l1loss(A2B2A, xA) + l1loss(B2A2B, xB)
            kl = kl_for_vae(meanA, stdA) + kl_for_vae(meanB, stdB)

            # Save to Log
            lm.add_torch_stat("D_adv", D_adv)
            lm.add_torch_stat("G_adv", G_adv)
            lm.add_torch_stat("cyc", cyc)
            lm.add_torch_stat("KL", kl)

    # Print total log of train phase
    print("Eval > \n\t", end='')
    lm.print_stat()
    # Save per defined epoch
    if int(epoch % save_per_epoch) == 0:
        parm_path = model_dir+"/parm/"+str(epoch)+".pt"
        torch.save(VAE_E.state_dict(), parm_path)
