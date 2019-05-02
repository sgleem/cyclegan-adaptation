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
parser.add_argument("--clean_dir", default="data/timit/target", type=str)
parser.add_argument("--model_dir", default="model/cyclegan", type=str)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--size", default=1, type=int)
args = parser.parse_args()
#####################################################################
adapt_dir = args.adapt_dir
dev_dir = args.dev_dir
clean_dir = args.clean_dir
model_dir = args.model_dir
#####################################################################
epochs = 1000
batch_size = 64
lr_g = 0.00002
lr_d = 0.00001

change_epoch = 200
save_per_epoch = 5
adv_coef = 1.0; cyc_coef = 10.0; id_coef = 5.0
#####################################################################
torch.cuda.empty_cache()
os.system("mkdir -p "+ model_dir +"/parm")

adapt_storage = read_feat(adapt_dir+"/feats.ark", delta=True)
dev_storage = read_feat(dev_dir+"/feats.ark", delta=True)
clean_storage = read_feat(clean_dir+"/feats.ark", delta=True)

for dataset in [adapt_storage, dev_storage, clean_storage]:
    for utt_id, feat_mat in dataset.items():
        feat_mat = pp.matrix_normalize(feat_mat, axis=1, fcn_type="mean")
        dataset[utt_id] = feat_mat

adapt_set = pp.make_cnn_dataset(adapt_storage, input_size=128); print(len(adapt_set))
dev_set = pp.make_cnn_dataset(dev_storage, input_size=128); print(len(dev_set))
clean_set = pp.make_cnn_dataset(clean_storage, input_size=128); print(len(clean_set))

adapt_loader = DataLoader(adapt_set, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=batch_size)
clean_loader = DataLoader(clean_set, batch_size=batch_size, shuffle=True)

print(len(adapt_loader))
print(len(dev_loader))
print(len(clean_loader))

# Gn2c = net.Generator_CNN(feat_dim=120)
# Gc2n = net.Generator_CNN(feat_dim=120)
# Dc = net.Discriminator(feat_dim=120, hidden_dim=512)
# Dn = net.Discriminator(feat_dim=120, hidden_dim=512)

Gn2c = net.Generator_CNN(feat_dim=120, num_down=2, num_res=6, num_up=2)
Gc2n = net.Generator_CNN(feat_dim=120, num_down=2, num_res=6, num_up=2)
Dc = net.Discriminator_CNN(feat_dim=120, hidden_dim=512)
Dn = net.Discriminator_CNN(feat_dim=120, hidden_dim=512)

Gn2c_opt = optim.Adam(Gn2c.parameters(), lr=lr_g)
Gc2n_opt = optim.Adam(Gc2n.parameters(), lr=lr_g)
Dc_opt = optim.Adam(Dc.parameters(), lr=lr_d)
Dn_opt = optim.Adam(Dn.parameters(), lr=lr_d)

Gn2c_sch = scheduler.StepLR(Gn2c_opt, step_size=100, gamma=0.5)
Gc2n_sch = scheduler.StepLR(Gc2n_opt, step_size=100, gamma=0.5)
Dc_sch = scheduler.StepLR(Dc_opt, step_size=100, gamma=0.5)
Dn_sch = scheduler.StepLR(Dn_opt, step_size=100, gamma=0.5)

for model in [Gn2c, Gc2n, Dc, Dn]:
    model.cuda()

lm = LogManager()
for stype in ["D_adv", "G_adv", "cyc", "id"]:
    lm.alloc_stat_type(stype)

torch.save(Gn2c, model_dir+"/init.pt")

for epoch in range(epochs):
    print("EPOCH :", epoch)
    # coefficient change
    if epoch >= change_epoch:
        id_coef = 0.0
        for sch in [Gn2c_sch, Gc2n_sch, Dc_sch, Dn_sch]:
            sch.step()
    # Train
    lm.init_stat()
    for model in [Gn2c, Gc2n, Dc, Dn]:
        model.train()
    # D & SPK phase
    for noise_utt, clean_utt in zip(adapt_loader, clean_loader):
        n = noise_utt.cuda().float()
        c = clean_utt.cuda().float()

        n2c = Gn2c(n)
        c2n = Gc2n(c)
        # Discriminator training
        clean_true = Dc(c); clean_false = Dc(n2c)
        noise_true = Dn(n); noise_false = Dn(c2n)

        adv1 = l2loss(clean_true, 1) / 2 + l2loss(noise_true, 1) / 2
        adv2 = l2loss(clean_false, 0) / 2 + l2loss(noise_false, 0) / 2
        adv = adv_coef * (adv1 + adv2)

        # Update Parameter
        total = adv
        for opt in [Dc_opt, Dn_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [Dc_opt, Dn_opt]:
            opt.step()

        # Save to Log
        lm.add_torch_stat("D_adv", adv)
    # G phase
    for noise_utt, clean_utt in zip(adapt_loader, clean_loader):
        n = noise_utt.cuda().float()
        c = clean_utt.cuda().float()

        n2c = Gn2c(n)
        c2n = Gc2n(c)
        # Adversarial training
        clean_false = Dc(n2c)
        noise_false = Dn(c2n)

        adv1 = l2loss(clean_false, 1) / 2
        adv2 = l2loss(noise_false, 1) / 2
        adv = adv_coef * (adv1 + adv2)

        # Cycle consistent training
        n2c2n = Gc2n(n2c); c2n2c = Gn2c(c2n)
        cyc1 = l2loss(n2c2n, n); cyc2 = l2loss(c2n2c, c)
        cyc = cyc_coef * (cyc1 + cyc2)

        # Id traning
        total = adv + cyc
        if id_coef > 0.0:
            c2c = Gn2c(c); n2n = Gc2n(n)
            idl1 = l2loss(c2c, c); idl2 = l2loss(n2n, n)
            idl = id_coef * (idl1 + idl2)
            total += idl
        
        # Update Parameter
        for opt in [Gn2c_opt, Gc2n_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [Gn2c_opt, Gc2n_opt]:
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
    for model in [Gn2c, Gc2n, Dc, Dn]:
        model.eval()
    with torch.no_grad():
        # Accumulate adv, cyc, id loss
        for dev_utt in dev_loader:
            n = dev_utt.cuda().float()

            n2c = Gn2c(n)
            # Adversarial stat
            clean_false = Dc(n2c)

            D_adv = adv_coef * l2loss(clean_false, 0)
            G_adv = adv_coef * l2loss(clean_false, 1)

            # Cycle consistent stat
            n2c2n = Gc2n(n2c)
            cyc = cyc_coef * l2loss(n2c2n, n)

            # Id stat
            if id_coef > 0.0:
                n2n = Gc2n(n)
                idl = id_coef * l2loss(n2n, n)

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
        torch.save(Gn2c.state_dict(), parm_path)
