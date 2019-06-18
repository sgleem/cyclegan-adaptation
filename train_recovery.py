import os
import sys
import argparse

import net
import data_preprocess as pp
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as sched


from tool.log import LogManager
from tool.loss import bcloss, l2loss
from tool.kaldi.kaldi_manager import read_feat, read_vec

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--adapt_dir", default="data/ntimit_30%zero_10ms/adapt", type=str)
parser.add_argument("--dev_dir", default="data/ntimit_30%zero_10ms/dev", type=str)
parser.add_argument("--target_dir", default="data/ntimit/target", type=str)
parser.add_argument("--model_dir", default="model/cyclegan_recovery", type=str)
args = parser.parse_args()
#####################################################################
adapt_dir = args.adapt_dir
dev_dir = args.dev_dir
target_dir = args.target_dir
model_dir = args.model_dir
#####################################################################
packet_epochs = 10
epochs = 1000
batch_size = 64
lr_packet = 0.0001
lr_g = 0.0002
lr_d = 0.0001

change_epoch = 200
save_per_epoch = 5
adv_coef = 1.0; cyc_coef = 10.0; id_coef = 5.0

utt_cmvn=False; spk_cmvn=False; delta=True
#####################################################################
torch.cuda.empty_cache()
os.system("mkdir -p "+ model_dir +"/parm")


adapt_storage = read_feat(adapt_dir+"/feats.ark", utt_cmvn=utt_cmvn, spk_cmvn=spk_cmvn, delta=delta)
dev_storage = read_feat(dev_dir+"/feats.ark", utt_cmvn=utt_cmvn, spk_cmvn=spk_cmvn, delta=delta)
target_storage = read_feat(target_dir+"/feats.ark", utt_cmvn=utt_cmvn, spk_cmvn=spk_cmvn, delta=delta)

adapt_lab = read_vec(adapt_dir+"/packet.ark")
dev_lab = read_vec(dev_dir+"/packet.ark")

adapt_set = pp.make_cnn_dataset_and_lab(adapt_storage, adapt_lab, frame_size=128, step_size=64)
dev_set = pp.make_cnn_dataset_and_lab(dev_storage, dev_lab, frame_size=128, step_size=64)
target_set = pp.make_cnn_dataset(target_storage, frame_size=128, step_size=64)

adapt_loader = DataLoader(adapt_set, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dev_set, batch_size=batch_size)
target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False)

G_packet = net.Discriminator_CNN(feat_dim=120, hidden_dim=512)
Gn2c = net.Generator_CNN(feat_dim=120)
Gc2n = net.Generator_CNN(feat_dim=120)
# G_packet for Gc2n is Xn's packet label
Dn = net.Discriminator_CNN(feat_dim=120, hidden_dim=512)
Dc = net.Discriminator_CNN(feat_dim=120, hidden_dim=512)

torch.save(G_packet, model_dir+"/init_packet.pt")
torch.save(Gn2c, model_dir+"/init_G.pt")

G_packet_opt = optim.Adam(G_packet.parameters(), lr=lr_packet)
Gn2c_opt = optim.Adam(Gn2c.parameters(), lr=lr_g)
Gc2n_opt = optim.Adam(Gc2n.parameters(), lr=lr_g)
Dn_opt = optim.Adam(Dn.parameters(), lr=lr_d)
Dc_opt = optim.Adam(Dc.parameters(), lr=lr_d)

G_packet_sch = sched.ReduceLROnPlateau(G_packet_opt, factor=0.5, patience=1, verbose=True)
Gn2c_sch = sched.StepLR(Gn2c_opt, step_size=100, gamma=0.5)
Gc2n_sch = sched.StepLR(Gc2n_opt, step_size=100, gamma=0.5)
Dn_sch = sched.StepLR(Dn_opt, step_size=100, gamma=0.5)
Dc_sch = sched.StepLR(Dc_opt, step_size=100, gamma=0.5)

lm = LogManager()
lm.alloc_stat_type_list(["train_loss" ,"dev_loss"])

G_packet.cuda()
print("Train packet loss discrimination network")
for epoch in range(packet_epochs):
    print("EPOCH :", epoch)
    G_packet.train()
    lm.init_stat()
    for frame_mat, packet_lab in adapt_loader:
        x = torch.Tensor(frame_mat).cuda().float()
        y = torch.Tensor(packet_lab).cuda().float()

        pred = G_packet(x)
        loss = bcloss(pred.squeeze(dim=2), y)

        G_packet_opt.zero_grad()
        loss.backward()
        G_packet_opt.step()

        lm.add_torch_stat("train_loss", loss)
    G_packet.eval()
    with torch.no_grad():
        for frame_mat, packet_lab in dev_loader:
            x = torch.Tensor(frame_mat).cuda().float()
            y = torch.Tensor(packet_lab).cuda().float()

            pred = G_packet(x) 
            loss = bcloss(pred.squeeze(dim=2), y)

            lm.add_torch_stat("dev_loss", loss)
    dev_loss = lm.get_stat("dev_loss")
    G_packet_sch.step(dev_loss)

    lm.print_stat()
    parm_path = model_dir+"/parm/"+str(epoch)+"_packet.pt"
    torch.save(G_packet.state_dict(), parm_path)

print("Train cycle-consitent network")
# load to gpu
for model in [Gn2c, Gc2n, Dn, Dc]:
    model.cuda()

lm = LogManager()
lm.alloc_stat_type_list(["D_adv", "G_adv", "cyc", "id"])

for epoch in range(epochs):
    print("EPOCH :", epoch)
    # coefficient change
    if epoch >= change_epoch:
        id_coef = 0.0
        for sch in [Gn2c_sch, Gc2n_sch, Dn_sch, Dc_sch]:
            sch.step()
    
    # load to gpu
    for model in [Gn2c, Gc2n, Dn, Dc]:
        model.train()
    
    # Discriminator
    for n_zip, c in zip(adapt_loader, target_loader):
        n = n_zip[0]; n_lab = n_zip[1]
        n = torch.Tensor(n).cuda().float()
        n_lab = torch.Tensor(n_lab).cuda().float().unsqueeze(dim=2)
        c = torch.Tensor(c).cuda().float()
    
        n_recover = Gn2c(n)
        n_pred = G_packet(n)
        n2c = n_pred * n + (1-n_pred) * n_recover

        c_drop = Gc2n(c)
        c_pred = torch.sigmoid(torch.rand_like(c)).cuda().float()
        c2n = c_pred * c + (1-c_pred) * c_drop

        clean_true = Dc(c); clean_false = Dc(n2c)
        noise_true = Dn(n); noise_false = Dn(c2n)

        adv_t = l2loss(clean_true, 1) / 2 + l2loss(noise_true, 1) / 2
        adv_f = l2loss(clean_false, 0) / 2 + l2loss(noise_false, 0) / 2
        adv = adv_f + adv_t

        # Update Parameter
        total = adv_coef * adv
        for opt in [Dc_opt, Dn_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [Dc_opt, Dn_opt]:
            opt.step()

        # Save to Log
        lm.add_torch_stat("D_adv", adv)
    
    # Generator
    for n_zip, c in zip(adapt_loader, target_loader):
        n = n_zip[0]; n_lab = n_zip[1]
        n = torch.Tensor(n).cuda().float()
        n_lab = torch.Tensor(n_lab).cuda().float().unsqueeze(dim=2)
        c = torch.Tensor(c).cuda().float()

        # Adversarial learning
        n_recover = Gn2c(n)
        n_pred = G_packet(n)
        n2c = n_pred * n + (1-n_pred) * n_recover

        c_drop = Gc2n(c)
        c_pred = torch.sigmoid(torch.rand_like(c)).cuda().float()
        c2n = c_pred * c + (1-c_pred) * c_drop

        clean_false = Dc(n2c)
        noise_false = Dn(c2n)

        adv_f = l2loss(clean_false, 1) / 2 + l2loss(noise_false, 1) / 2
        adv = adv_f

        # Cycle consistent learning
        n2c_drop = Gc2n(n2c)
        n2c2n = n_pred * n2c + (1-n_pred) * n2c_drop
        
        c2n_recover = Gn2c(c2n)
        c2n2c = c_pred * c2n + (1-c_pred) * c2n_recover

        cyc1 = l2loss(n2c2n, n); cyc2 = l2loss(c2n2c, c)
        cyc = cyc1 + cyc2
        
        # calculate total loss
        total = (adv_coef * adv) + (cyc_coef * cyc)

        # Id traning
        if id_coef > 0.0:
            n2n = Gc2n(n)
            n2n = n_lab * n + (1-n_lab) * n2n

            c2c = Gn2c(c); c_pred = G_packet(c)
            c2c = c_pred * c + (1-c_pred) * c2c
            idl1 = l2loss(c2c, c); idl2 = l2loss(n2n, n)
            idl =  idl1 + idl2
            total += (id_coef * idl)

        # Update Parameter
        for opt in [Gn2c_opt, Gc2n_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [Gn2c_opt, Gc2n_opt]:
            opt.step()

        # Save to Log
        lm.add_torch_stat("G_adv", adv)
        lm.add_torch_stat("cyc", cyc)
        lm.add_torch_stat("id", idl)
    # Print total log of train phase
    print("Train G > \n\t", end='')
    lm.print_stat()
    # Eval
    lm.init_stat()
    for model in [Gn2c, Gc2n, Dc, Dn]:
        model.eval()
    with torch.no_grad():
        for n, n_lab in dev_loader:
            n = torch.Tensor(n).cuda().float()
            n_lab = torch.Tensor(n_lab).cuda().float().unsqueeze(dim=2)
            
            # Adversarial learning
            n_recover = Gn2c(n)
            n_pred = G_packet(n)
            n2c = n_pred * n + (1-n_pred) * n_recover

            clean_false = Dc(n2c)
            noise_true = Dn(n)
         
            # Cycle consistent learning
            n2c_drop = Gc2n(n2c)
            n2c2n = n_pred * n2c + (1-n_pred) * n2c_drop

            # Calc loss
            D_adv = l2loss(clean_false, 0) / 2 + l2loss(noise_true, 1) / 2
            G_adv = l2loss(clean_false, 1)
            cyc = l2loss(n2c2n, n)

            # Id traning
            if id_coef > 0.0:
                n2n = Gc2n(n)
                n2n = n_lab * n + (1-n_lab) * n2n
                idl = l2loss(n2n, n)

            # Save to Log
            lm.add_torch_stat("D_adv", D_adv)
            lm.add_torch_stat("G_adv", G_adv)
            lm.add_torch_stat("cyc", cyc)
            lm.add_torch_stat("id", idl)

    # Print total log of train phase
    print("Eval > \n\t", end='')
    lm.print_stat()
    # Save per defined epoch
    if int(epoch % save_per_epoch) == 0:
        parm_path = model_dir+"/parm/"+str(epoch)+"_G.pt"
        torch.save(Gn2c.state_dict(), parm_path)



        