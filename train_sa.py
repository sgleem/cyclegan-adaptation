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
from tool.loss import nllloss, l2loss, calc_err
from tool.kaldi.kaldi_manager import read_feat

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--adapt_dir", default="data/ntimit/adapt", type=str)
parser.add_argument("--dev_dir", default="data/ntimit/dev", type=str)
parser.add_argument("--wide_dir", default="data/timit/target", type=str)
parser.add_argument("--model_dir", default="model/cyclegan", type=str)
parser.add_argument("--loss_per", default=0, type=int)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--size", default=1, type=int)
args = parser.parse_args()
#####################################################################
adapt_dir = args.adapt_dir
dev_dir = args.dev_dir
wide_dir = args.wide_dir
model_dir = args.model_dir
loss_per = args.loss_per
#####################################################################
epochs = 1000
batch_size = 64
lr_g = 0.0002
lr_d = 0.0001

change_epoch = 200
save_per_epoch = 5
adv_coef = 1.0; cyc_coef = 10.0; id_coef = 5.0
#####################################################################
torch.cuda.empty_cache()
os.system("mkdir -p "+ model_dir +"/parm")

adapt_storage = read_feat(adapt_dir+"/feats.ark", utt_cmvn=True, delta=True)
dev_storage = read_feat(dev_dir+"/feats.ark", utt_cmvn=True, delta=True)
wide_storage = read_feat(wide_dir+"/feats.ark", utt_cmvn=True, delta=True)

adapt_set = pp.make_cnn_dataset(adapt_storage, input_size=128, step_size=64); print(len(adapt_set))
dev_set = pp.make_cnn_dataset(dev_storage, input_size=128, step_size=64); print(len(dev_set))
wide_set = pp.make_cnn_dataset(wide_storage, input_size=128, step_size=64); print(len(wide_set))

adapt_loader = DataLoader(adapt_set, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=batch_size)
wide_loader = DataLoader(wide_set, batch_size=batch_size, shuffle=True)

Gn2w = net.Generator_CNN(feat_dim=120)
Gw2n = net.Generator_CNN(feat_dim=120)
Dn = net.Discriminator_CNN(feat_dim=120, hidden_dim=512)
Dw = net.Discriminator_CNN(feat_dim=120, hidden_dim=512)

Gn2w_opt = optim.Adam(Gn2w.parameters(), lr=lr_g)
Gw2n_opt = optim.Adam(Gw2n.parameters(), lr=lr_g)
Dn_opt = optim.Adam(Dn.parameters(), lr=lr_d)
Dw_opt = optim.Adam(Dw.parameters(), lr=lr_d)

Gn2w_sch = scheduler.StepLR(Gn2w_opt, step_size=100, gamma=0.5)
Gw2n_sch = scheduler.StepLR(Gw2n_opt, step_size=100, gamma=0.5)
Dn_sch = scheduler.StepLR(Dn_opt, step_size=100, gamma=0.5)
Dw_sch = scheduler.StepLR(Dw_opt, step_size=100, gamma=0.5)

for model in [Gn2w, Gw2n, Dn, Dw]:
    model.cuda()

lm = LogManager()
for stype in ["D_adv", "G_adv", "cyc", "id"]:
    lm.alloc_stat_type(stype)

torch.save(Gn2w, model_dir+"/init.pt")

for epoch in range(epochs):
    print("EPOCH :", epoch)
    # coefficient change
    if epoch >= change_epoch:
        id_coef = 0.0
        for sch in [Gn2w_sch, Gw2n_sch, Dn_sch, Dw_sch]:
            sch.step()
    # Train
    lm.init_stat()
    for model in [Gn2w, Gw2n, Dn, Dw]:
        model.train()
    # D & SPK phase
    for narrow_utt, wide_utt in zip(adapt_loader, wide_loader):
        n = torch.Tensor(narrow_utt).cuda().float()
        w = torch.Tensor(wide_utt).cuda().float()

        n2w = Gn2w(n)
        w2n = Gw2n(w)
        # Discriminator training
        wide_true = Dw(w); wide_false = Dw(n2w)
        narrow_true = Dn(n); narrow_false = Dn(w2n)

        adv_t = l2loss(wide_true, 1) / 2 + l2loss(narrow_true, 1) / 2
        adv_f = l2loss(wide_false, 0) / 2 + l2loss(narrow_false, 0) / 2
        adv = adv_f + adv_t

        # Update Parameter
        total = adv_coef * adv
        for opt in [Dw_opt, Dn_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [Dw_opt, Dn_opt]:
            opt.step()

        # Save to Log
        lm.add_torch_stat("D_adv", adv)
    # G phase
    for narrow_utt, wide_utt in zip(adapt_loader, wide_loader):
        n = torch.Tensor(narrow_utt).cuda().float()
        w = torch.Tensor(wide_utt).cuda().float()

        n2w = Gn2w(n)
        w2n = Gw2n(w)
        # Adversarial training
        wide_false = Dw(n2w)
        narrow_false = Dn(w2n)

        adv_f = l2loss(wide_false, 1) / 2 + l2loss(narrow_false, 1) / 2
        adv = adv_f

        # Cycle consistent training
        n2w2n = Gw2n(n2w); w2n2w = Gn2w(w2n)
        cyc1 = l2loss(n2w2n, n); cyc2 = l2loss(w2n2w, w)
        cyc = cyc1 + cyc2

        # Id traning
        total = (adv_coef * adv) + (cyc_coef * cyc)
        if id_coef > 0.0:
            w2w = Gn2w(w); n2n = Gw2n(n)
            idl1 = l2loss(w2w, w); idl2 = l2loss(n2n, n)
            idl =  idl1 + idl2
            total += (id_coef * idl)
        
        # Update Parameter
        for opt in [Gn2w_opt, Gw2n_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [Gn2w_opt, Gw2n_opt]:
            opt.step()
        
        # Save to Log
        lm.add_torch_stat("G_adv", adv)
        lm.add_torch_stat("cyc", cyc)
        if id_coef > 0.0:
            lm.add_torch_stat("id", idl)
    # Print total log of train phase
    print("Train G > \n\t", end='')
    lm.print_stat()
    # Eval
    lm.init_stat()
    for model in [Gn2w, Gw2n, Dw, Dn]:
        model.eval()
    with torch.no_grad():
        # Accumulate adv, cyc, id loss
        for dev_utt in dev_loader:
            n = torch.Tensor(dev_utt).cuda().float()

            n2w = Gn2w(n)
            wide_false = Dw(n2w); noise_true = Dn(n)

            D_adv = l2loss(wide_false, 0) / 2 + l2loss(noise_true, 1) / 2
            G_adv = l2loss(wide_false, 1)

            # Cycle consistent stat
            n2w2n = Gw2n(n2w)
            cyc = l2loss(n2w2n, n)

            # Id stat
            if id_coef > 0.0:
                n2n = Gw2n(n)
                idl = l2loss(n2n, n)

            # Save to Log
            lm.add_torch_stat("D_adv", D_adv)
            lm.add_torch_stat("G_adv", G_adv)
            lm.add_torch_stat("cyc", cyc)
            if id_coef > 0.0:
                lm.add_torch_stat("id", idl)
    # Print total log of train phase
    print("Eval > \n\t", end='')
    lm.print_stat()
    # Save per defined epoch
    if int(epoch % save_per_epoch) == 0:
        parm_path = model_dir+"/parm/"+str(epoch)+".pt"
        torch.save(Gn2w.state_dict(), parm_path)
