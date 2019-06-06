import os
import argparse
import random
import pickle as pk
import numpy as np

import torch
import torch.optim as opt
import torch.optim.lr_scheduler as sch

from net import *
from module import *
from data_preprocess import matrix_normalize
from tool.log import LogManager
from tool.loss import nllloss, calc_err
from tool.kaldi import kaldi_io as kio
from tool.kaldi.kaldi_manager import read_feat, read_ali, read_vec
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--train_feat_dir", default="data/timit/train", type=str)
parser.add_argument("--train_ali_dir", default="ali/timit/train", type=str)
parser.add_argument("--dev_feat_dir", default="data/timit/dev", type=str)
parser.add_argument("--dev_ali_dir", default="ali/timit/dev", type=str)
parser.add_argument("--si_dir", default="model/gru_norm", type=str)
parser.add_argument("--sa_dir", default="model/vae_timit", type=str)
parser.add_argument("--resi_dir", default="model/gru_retrained", type=str)
args = parser.parse_args()
#####################################################################
train_feat_dir = args.train_feat_dir
train_ali_dir = args.train_ali_dir
dev_feat_dir = args.dev_feat_dir
dev_ali_dir = args.dev_ali_dir
si_dir = args.si_dir
sa_dir = args.sa_dir
resi_dir = args.resi_dir
#####################################################################
epochs = 24
lr = 0.0001
#####################################################################

os.system("mkdir -p " + resi_dir + "/parm")
os.system("mkdir -p " + resi_dir + "/opt")

train_feat = read_feat(train_feat_dir+"/feats.ark", cmvn=True, delta=True)
dev_feat = read_feat(dev_feat_dir+"/feats.ark", cmvn=True, delta=True)
train_ali = read_ali(train_ali_dir+"/ali.*.gz", train_ali_dir+"/final.mdl")
dev_ali = read_ali(dev_ali_dir+"/ali.*.gz", dev_ali_dir+"/final.mdl")
train_utt = list(train_feat.keys())
dev_utt = list(dev_feat.keys())
train_ivecs = read_vec(train_feat_dir+"/ivectors.ark")
dev_ivecs = read_vec(dev_feat_dir+"/ivectors.ark")

model_sa = torch.load(sa_dir+"/init.pt")
model_sa.load_state_dict(torch.load(sa_dir+"/final.pt"))
model_sa.cuda()

# get normalized feat
print("Feature generation start")
dset_dict={"train": train_feat, "dev":dev_feat}
ivec_dict={"train": train_ivecs, "dev":dev_ivecs}

model_sa.eval()
with torch.no_grad():
    for dset_name in ["train", "dev"]:
        dataset = dset_dict[dset_name]
        ivecset = ivec_dict[dset_name]
        for utt_id, feat_mat in dataset.items():
            x = torch.Tensor([feat_mat]).cuda().float()

            spk_id = utt_id.split("_")[0]
            ivecs = ivecset[spk_id]
            ivecs = torch.Tensor([ivecs]).cuda().float()
            print(x.size())
            print(ivecs.size())
            
            x = model_sa(x,ivecs)
            x = x[0].cpu().detach().numpy()
            dataset[utt_id] = x
print("Feature generation complete")

model_si = torch.load(si_dir+"/init.pt")
model_si.load_state_dict(torch.load(si_dir+"/final.pt"))
model_si.cuda()

model_opt = opt.Adam(model_si.parameters(), lr=lr)
model_sch = sch.ReduceLROnPlateau(model_opt, factor=0.5, patience=0, verbose=True)
lm = LogManager()
lm.alloc_stat_type_list(["train_loss","train_acc","dev_loss","dev_acc"])

# model training
torch.save(model_si, resi_dir+"/init.pt")
for epoch in range(epochs):
    print("Epoch",epoch)
    random.shuffle(train_utt)
    model_si.train()
    lm.init_stat()
    for utt_id in train_utt:
        x = train_feat[utt_id]
        ali = train_ali.get(utt_id,[])
        if len(ali) == 0:
            continue

        x = torch.Tensor(x).cuda().float()
        y = torch.Tensor(ali).cuda().long()

        pred = model_si(x)
        loss = nllloss(pred, y)
        err = calc_err(pred, y)

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()

        lm.add_torch_stat("train_loss", loss)
        lm.add_torch_stat("train_acc", 1.0 - err)

    model_si.eval()
    with torch.no_grad():
        for utt_id in dev_utt:
            x = dev_feat[utt_id]
            ali = dev_ali[utt_id]

            x = torch.Tensor(x).cuda().float()
            y = torch.Tensor(ali).cuda().long()

            pred = model_si(x)
            loss = nllloss(pred, y)
            err = calc_err(pred, y)

            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", 1.0 - err)
    dev_loss = lm.get_stat("dev_loss")
    model_sch.step(dev_loss)

    lm.print_stat()

    parm_path = resi_dir+"/parm/"+str(epoch)+".pt"
    torch.save(model_si.state_dict(), parm_path)

    opt_path = resi_dir+"/opt/"+str(epoch)+".pt"
    torch.save(model_opt.state_dict(), opt_path)
