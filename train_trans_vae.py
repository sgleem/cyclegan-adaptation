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
parser.add_argument("--model_dir", default="model/vae_timit_0603", type=str)
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

save_per_epoch = 5
adv_coef = 1.0; spk_coef = 10.0; kl_coef = 0.001; cyc_coef=10.0; norm_coef = 0.001; recon_coef = 10.0
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

VAE_E = net.VariationalEncoder(feat_dim=120)
mTranslator = net.MeanTranslator(feat_dim=120, ivec_dim=100)
VAE_G = net.VariationalDecoder(feat_dim=120)
VAE_D = net.VAE_Discriminator(feat_dim=120, hidden_dim=512, spk_dim=spk_dim)

E_opt = optim.Adam(VAE_E.parameters(), lr=lr_e)
m_opt = optim.Adam(mTranslator.parameters(), lr=lr_g)
G_opt = optim.Adam(VAE_G.parameters(), lr=lr_g)
D_opt = optim.Adam(VAE_D.parameters(), lr=lr_d)

E_sch = scheduler.StepLR(E_opt, step_size=100, gamma=0.5)
m_sch = scheduler.StepLR(m_opt, step_size=100, gamma=0.5)
G_sch = scheduler.StepLR(G_opt, step_size=100, gamma=0.5)
D_sch = scheduler.StepLR(D_opt, step_size=100, gamma=0.5)

for model in [VAE_E, mTranslator, VAE_G, VAE_D]:
    model.cuda()
torch.save(VAE_E, model_dir+"/init.pt")

# Train start
print("Train start")
lm = LogManager()
for stype in ["D_adv", "D_spk", "G_adv", "G_spk", "norm", "recon", "cyc", "KL", "content"]:
    lm.alloc_stat_type(stype)
for epoch in range(epochs):
    print("EPOCH :", epoch)
    
    # Domain Translation - Discriminator
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

        mxA, sxA = VAE_E(xA)
        mxB, sxB = VAE_E(xB)

        mxA2B, siA = mTranslator(mxA, iA, iB)
        mxB2A, siB = mTranslator(mxB, iB, iA)

        xA2B = VAE_G(mxA2B, sxA)
        xB2A = VAE_G(mxB2A, sxB)

        tf_A, spk_A = VAE_D(xA)
        tf_B, spk_B = VAE_D(xB)
        tf_A2B, spk_A2B = VAE_D(xA2B)
        tf_B2A, spk_B2A = VAE_D(xB2A)

        D_adv1 = torch.mean(tf_A2B) - torch.mean(tf_A)
        D_adv2 = torch.mean(tf_B2A) - torch.mean(tf_B)
        D_adv = D_adv1 + D_adv2

        D_spk1 = nllloss(spk_A, tA) + nllloss(spk_B, tB) 
        D_spk2 = nllloss(spk_A2B, tA) + nllloss(spk_B2A, tB)
        D_spk = D_spk1 + D_spk2

        total = adv_coef * D_adv + spk_coef * D_spk

        for opt in [D_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [D_opt]:
            opt.step()
        # Save to Log
        lm.add_torch_stat("D_adv", D_adv)
        lm.add_torch_stat("D_spk", D_spk)

    # Reconstruction
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

        mxA, stdA = VAE_E(xA)
        mxB, stdB = VAE_E(xB)

        xA2A = VAE_G(mxA, stdA)
        xB2B = VAE_G(mxB, stdB)

        # _, spk_siA = VAE_D(siA)
        # _, spk_siB = VAE_D(siB)

        norm = 0 #- nllloss(spk_siA, tA) - nllloss(spk_siB, tB)
        recon = l1loss(xA2A, xA) + l1loss(xB2B, xB)
        kl = kl_for_vae(mxA, stdA) + kl_for_vae(mxB, stdB)

        total = norm_coef * norm + recon_coef * recon + kl_coef * kl

        for opt in [E_opt, G_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [E_opt, G_opt]:
            opt.step()
        # Save to Log
        # lm.add_torch_stat("norm", norm)
        lm.add_torch_stat("recon", recon)
        lm.add_torch_stat("KL", kl)


    # Domain Translation - Generator
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

        mxA, stdA = VAE_E(xA)
        mxB, stdB = VAE_E(xB)

        mxA2B, siA = mTranslator(mxA, iA, iB)
        mxB2A, siB = mTranslator(mxB, iB, iA)

        xA2B = VAE_G(mxA2B, stdA)
        xB2A = VAE_G(mxB2A, stdB)

        tf_A, _ = VAE_D(xA)
        tf_B, _ = VAE_D(xB)
        tf_A2B, spk_A2B = VAE_D(xA2B)
        tf_B2A, spk_B2A = VAE_D(xB2A)

        G_adv1 = torch.mean(tf_A) - torch.mean(tf_A2B)
        G_adv2 = torch.mean(tf_B) - torch.mean(tf_B2A)
        G_adv = G_adv1 + G_adv2

        G_spk = nllloss(spk_A2B, tB) + nllloss(spk_B2A, tA) 

        mxA2B, stdA2B = VAE_E(xA2B)
        mxB2A, stdB2A = VAE_E(xB2A)

        mxA2B2A, siA2B = mTranslator(mxA2B, iB, iA)
        mxB2A2B, siB2A = mTranslator(mxB2A, iA, iB)

        xA2B2A = VAE_G(mxA2B2A, stdA2B)
        xB2A2B = VAE_G(mxB2A2B, stdB2A)

        cyc = l1loss(xA2B2A, xA) + l1loss(xB2A2B, xB)
        content = l1loss(siA2B, siA) + l1loss(siB2A, siB)
        kl = kl_for_vae(mxA, stdA) + kl_for_vae(mxB, stdB) + kl_for_vae(mxB2A, stdB2A) + kl_for_vae(mxA2B, stdA2B)
        
        total = adv_coef * G_adv + spk_coef * G_spk + kl_coef * kl + cyc_coef * cyc + content 
        for opt in [E_opt, m_opt, G_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [E_opt, m_opt, G_opt]:
            opt.step()

        # Save to Log
        lm.add_torch_stat("G_adv", G_adv)
        lm.add_torch_stat("G_spk", G_spk)
        lm.add_torch_stat("cyc", cyc)
        lm.add_torch_stat("content", content)
        lm.add_torch_stat("KL", kl)
    # Print total log of train phase
    print("Train G > \n\t", end='')
    lm.print_stat()
    # Eval
    lm.init_stat()
    for model in [VAE_E, mTranslator, VAE_G, VAE_D]:
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

            mxA, stdA = VAE_E(xA)
            mxB, stdB = VAE_E(xB)

            mxA2B, siA = mTranslator(mxA, iA, iB)
            mxB2A, siB = mTranslator(mxB, iB, iA)

            xA2B = VAE_G(mxA2B, stdA)
            xB2A = VAE_G(mxB2A, stdB)

            tf_A, _ = VAE_D(xA)
            tf_B, _ = VAE_D(xB)
            tf_A2B, _ = VAE_D(xA2B)
            tf_B2A, _ = VAE_D(xB2A)

            D_adv = torch.mean(tf_A2B) - torch.mean(tf_A) + torch.mean(tf_B2A) - torch.mean(tf_B)
            G_adv = -1 * D_adv

            mxA2B, stdA2B = VAE_E(xA2B)
            mxB2A, stdB2A = VAE_E(xB2A)

            mxA2B2A, siA2B = mTranslator(mxA2B, iB, iA)
            mxB2A2B, siB2A = mTranslator(mxB2A, iA, iB)

            xA2B2A = VAE_G(mxA2B2A, stdA2B)
            xB2A2B = VAE_G(mxB2A2B, stdB2A)

            cyc = l1loss(xA2B2A, xA) + l1loss(xB2A2B, xB)
            content = l1loss(siA2B, siA) + l1loss(siB2A, siB)
            kl = kl_for_vae(siA, stdA) + kl_for_vae(siB, stdB) + kl_for_vae(siB2A, stdB2A) + kl_for_vae(siA2B, stdA2B)

            # Save to Log
            lm.add_torch_stat("D_adv", D_adv)
            lm.add_torch_stat("G_adv", G_adv)
            lm.add_torch_stat("cyc", cyc)
            lm.add_torch_stat("content", content)
            lm.add_torch_stat("KL", kl)
    # Print total log of train phase
    print("Eval > \n\t", end='')
    lm.print_stat()
    # Save per defined epoch
    if int(epoch % save_per_epoch) == 0:
        parm_path = model_dir+"/parm/"+str(epoch)+".pt"
        torch.save(VAE_E.state_dict(), parm_path)


            

