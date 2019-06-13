import os
import sys
import argparse
import random

import numpy as np
import pickle as pk
import torch
import torch.optim as opt
import torch.optim.lr_scheduler as sch
from torch.nn.utils.rnn import pad_sequence

import net
import data_preprocess as pp
from tool.log import LogManager
from tool.loss import nllloss, calc_err
from tool.kaldi.kaldi_manager import read_feat, read_ali

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--train_feat_dir", default="data/timit/train", type=str)
parser.add_argument("--train_ali_dir", default="ali/timit_sgmm/train", type=str)
parser.add_argument("--dev_feat_dir", default="data/timit/dev", type=str)
parser.add_argument("--dev_ali_dir", default="ali/timit_sgmm/dev", type=str)
parser.add_argument("--model_dir", default="model/gru_WB", type=str)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--size", default=1, type=int)
args = parser.parse_args()
#####################################################################
train_feat_dir = args.train_feat_dir
train_ali_dir = args.train_ali_dir
dev_feat_dir = args.dev_feat_dir
dev_ali_dir = args.dev_ali_dir
model_dir = args.model_dir
#####################################################################
epochs = 22
batch_size = 1
lr = 0.0001
pdf_num = 5657 # 1920(timit tri3) 5626(ntimit NB) 5657(ntimit WB)
#####################################################################
os.system("mkdir -p " + model_dir + "/parm")
os.system("mkdir -p " + model_dir + "/opt")

train_feat = read_feat(train_feat_dir+"/feats.ark", delta=True)
dev_feat = read_feat(dev_feat_dir+"/feats.ark", delta=True)

train_ali = read_ali(train_ali_dir+"/ali.*.gz", train_ali_dir+"/final.mdl")
dev_ali = read_ali(dev_ali_dir+"/ali.*.gz", dev_ali_dir+"/final.mdl")

train_utt = list(train_feat.keys())
dev_utt = list(dev_feat.keys())

# sort by frame length
train_utt.sort(key=lambda x: len(train_feat[x]))
model = net.GRU_HMM(input_dim=120, hidden_dim=512, num_layers=4, output_dim=pdf_num)
torch.save(model, model_dir+"/init.pt")
model.cuda()
model_opt = opt.Adam(model.parameters(), lr=lr)
model_sch = sch.ReduceLROnPlateau(model_opt, factor=0.5, patience=0, verbose=True)

lm = LogManager()
lm.alloc_stat_type_list(["train_loss","train_acc","dev_loss","dev_acc"])

# prior calculation
prior = np.zeros(pdf_num)
for align in train_ali.values():
    for ali in align:
        prior[ali] += 1
prior /= np.sum(prior)
with open(model_dir+"/prior.pk", 'wb') as f:
    pk.dump(prior, f)

for storage in [train_feat, dev_feat]:
    for utt_id, feat_mat in storage.items():
        storage[utt_id] = pp.matrix_normalize(feat_mat, axis=1, fcn_type="mean")

# model training
for epoch in range(epochs):
    print("Epoch",epoch)
    random.shuffle(train_utt)
    model.train()
    lm.init_stat()
    # for utt_id in train_utt:
    for start_idx in range(0, len(train_utt), batch_size):
        cur_batch = train_utt[start_idx : min(start_idx + batch_size, len(train_utt))]
        x = []; y = []
        for utt_id in cur_batch:
            feat = train_feat[utt_id]
            ali = train_ali.get(utt_id,[])
            if len(ali) == 0:
                continue
            x.append(torch.Tensor(feat))
            y.append(torch.Tensor(ali))
        if len(x) == 0:
            continue
        x = pad_sequence(x)
        y = pad_sequence(y)
        y = torch.flatten(y)

        x = x.cuda().float()
        y = y.cuda().long()

        pred = model(x)
        pred = torch.flatten(pred, start_dim=0, end_dim=1)
        loss = nllloss(pred, y)
        err = calc_err(pred, y)

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()

        lm.add_torch_stat("train_loss", loss)
        lm.add_torch_stat("train_acc", 1.0 - err)
    model.eval()
    with torch.no_grad():
        for utt_id in dev_utt:
            x = dev_feat[utt_id]
            y = dev_ali.get(utt_id,[])
            if len(y) == 0:
                continue

            x = torch.Tensor(x).cuda().float()
            y = torch.Tensor(y).cuda().long()

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

