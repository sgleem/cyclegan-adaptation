import os
import sys
import argparse
import random

import numpy as np
import pickle as pk
import torch
import torch.optim as opt
import torch.optim.lr_scheduler as sch

import torch.distributed as dist

import net
import data_preprocess as pp
from tool.log import LogManager
from tool.loss import nllloss, calc_err
from tool.kaldi.kaldi_manager import read_feat, read_ali

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--train_feat_dir", default="data/wsj/train", type=str)
parser.add_argument("--train_ali_dir", default="ali/wsj/train", type=str)
parser.add_argument("--dev_feat_dir", default="data/wsj/dev", type=str)
parser.add_argument("--dev_ali_dir", default="ali/wsj/dev", type=str)
parser.add_argument("--model_dir", default="model/gru_wsj_norm", type=str)
parser.add_argument("--rank", "-r", default=0, type=int)
parser.add_argument("--size", "-s", default=1, type=int)
parser.add_argument("--master", "-m", default="speech102", type=str)
args = parser.parse_args()
#####################################################################
train_feat_dir = args.train_feat_dir
train_ali_dir = args.train_ali_dir
dev_feat_dir = args.dev_feat_dir
dev_ali_dir = args.dev_ali_dir
model_dir = args.model_dir

node_rank=args.rank
node_size=args.size
master_node=args.master
#####################################################################
epochs = 24
lr = 0.0001
pdf_num = 3424
#####################################################################
os.system("mkdir -p " + model_dir + "/parm")
os.system("mkdir -p " + model_dir + "/opt")
cwd="/home/lsg9311/cyclegan-adaptation/"
os.chdir(cwd)
print(os.getcwd())

train_feat = read_feat(train_feat_dir+"/feats.ark", delta=True)
dev_feat = read_feat(dev_feat_dir+"/feats.ark", delta=True)

print(len(train_feat))
print(len(dev_feat))

# for distributed
if node_size > 1:
    master_nfs="/nfs/"+master_node+"/hdd/lsg9311/cyclegan-adaptation/sharedfile"
    dist.init_process_group(backend="nccl", init_method="file://"+master_nfs, 
                            world_size=node_size, rank=node_rank)

train_ali = read_ali(train_ali_dir+"/ali.*.gz", train_ali_dir+"/final.mdl")
dev_ali = read_ali(dev_ali_dir+"/ali.*.gz", dev_ali_dir+"/final.mdl")

train_utt = list(train_feat.keys())
dev_utt = list(dev_feat.keys())

model = net.GRU_HMM(input_dim=120, hidden_dim=512, num_layers=5, output_dim=pdf_num)
torch.save(model, model_dir+"/init.pt")
model.cuda()
model_opt = opt.Adam(model.parameters(), lr=lr)
model_sch = sch.ReduceLROnPlateau(model_opt, factor=0.5, patience=0, verbose=True)

lm = LogManager()
lm.alloc_stat_type_list(["train_loss","train_acc","dev_loss","dev_acc"])

# preprocess
for dataset in [train_feat, dev_feat]:
    for utt_id, feat_mat in dataset.items():
        feat_mat = pp.matrix_normalize(feat_mat, axis=1, fcn_type="mean")
        dataset[utt_id] = feat_mat
# prior calculation
# prior = np.zeros(pdf_num)
# for align in train_ali.values():
#     for ali in align:
#         prior[ali] += 1
# prior /= np.sum(prior)
# with open(model_dir+"/prior.pk", 'wb') as f:
#     pk.dump(prior, f)
# model training
for epoch in range(epochs):
    print("Epoch",epoch)
    random.shuffle(train_utt)
    model.train()
    lm.init_stat()
    for utt_id in train_utt:
        x = train_feat[utt_id]
        # ali = train_ali[utt_id]
        ali = train_ali.get(utt_id,[])
        if len(ali) == 0:
            continue

        x = torch.Tensor(x).cuda().float()
        y = torch.Tensor(ali).cuda().long()

        pred = model(x)
        loss = nllloss(pred, y)
        err = calc_err(pred, y)

        model_opt.zero_grad()
        loss.backward()
        # for distributed
        if node_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= node_size
        model_opt.step()

        lm.add_torch_stat("train_loss", loss)
        lm.add_torch_stat("train_acc", 1.0 - err)
        
    model.eval()
    with torch.no_grad():
        for utt_id in dev_utt:
            x = dev_feat[utt_id]
            ali = dev_ali[utt_id]

            x = torch.Tensor(x).cuda().float()
            y = torch.Tensor(ali).cuda().long()

            pred = model(x)
            loss = nllloss(pred, y)
            err = calc_err(pred, y)

            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", 1.0 - err)
            
    dev_loss = lm.get_stat("dev_loss")
    model_sch.step(dev_loss)

    lm.print_stat()

    parm_path = model_dir+"/parm/"+str(epoch)+".pt"
    torch.save(model.state_dict(), parm_path)

    opt_path = model_dir+"/opt/"+str(epoch)+".pt"
    torch.save(model_opt.state_dict(), opt_path)

