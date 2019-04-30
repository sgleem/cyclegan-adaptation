import os
import sys
import random
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

import net
from tool.log import LogManager
from tool.loss import nllloss, calc_err
from preprocess import matrix_normalize
from tool.kaldi.kaldi_manager import read_feat
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--adapt_dir", default="data/ntimit/adapt", type=str)
parser.add_argument("--dev_dir", default="data/ntimit/dev", type=str)
parser.add_argument("--si_dir", default="model/gru_norm", type=str)
parser.add_argument("--sa_dir", default="model/fdlr", type=str)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--size", default=1, type=int)
args = parser.parse_args()
#####################################################################
adapt_dir = args.adapt_dir
dev_dir = args.dev_dir
si_dir = args.si_dir
sa_dir = args.sa_dir
#####################################################################
epochs = 50
lr = 0.0001
#####################################################################

#####################################################################
os.system("mkdir -p "+ sa_dir +"/parm")

adapt_feat = read_feat(adapt_dir+"/feats.ark", delta=True)
dev_feat = read_feat(dev_dir+"/feats.ark", delta=True)

adapt_utt = list(adapt_feat.keys())
dev_utt = list(dev_feat.keys())

for dataset in [adapt_feat, dev_feat]:
    for utt_id, feat_mat in dataset.items():
        feat_mat = matrix_normalize(feat_mat, axis=1, fcn_type="mean")
        feat_mat = torch.Tensor(feat_mat).cuda().float()
        dataset[utt_id] = feat_mat

model_si = torch.load(si_dir+"/init.pt")
model_si.load_state_dict(torch.load(si_dir+"/final.pt"))
model_si.cuda()
model_si.eval()

model_sa = net.FDLR(feat_dim=120, context_size=5)
torch.save(model_sa, sa_dir+"/init.pt")
model_sa.cuda()
model_opt = optim.Adam(model_sa.parameters(), lr=lr)
model_sch = scheduler.ReduceLROnPlateau(model_opt, factor=0.5, patience=0, verbose=True)

lm = LogManager()
lm.alloc_stat_type_list(["adapt_loss","adapt_acc","dev_loss","dev_acc"])

# extract alignment
adapt_ali = dict()
for utt_id in adapt_utt:
    x = adapt_feat[utt_id]
    y = model_si(x)
    ans = torch.max(y,dim=1)[1].long()
    adapt_ali[utt_id] = ans

dev_ali = dict()
for utt_id in dev_utt:
    x = dev_feat[utt_id]
    y = model_si(x)
    ans = torch.max(y,dim=1)[1].long()
    dev_ali[utt_id] = ans

# adaptation
for epoch in range(epochs):
    print("Epoch",epoch)
    random.shuffle(adapt_utt)
    for model in [model_si, model_sa]:
        model.train()
    lm.init_stat()
    for utt_id in adapt_utt:
        x = adapt_feat[utt_id]
        y = adapt_ali[utt_id]

        x = model_sa(x)
        pred = model_si(x)

        loss = nllloss(pred, y)
        err = calc_err(pred, y)

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()

        lm.add_torch_stat("adapt_loss", loss)
        lm.add_torch_stat("adapt_acc", 1.0 - err)

    for model in [model_si, model_sa]:
        model.eval()
    
    with torch.no_grad():
        for utt_id in dev_utt:
            x = dev_feat[utt_id]
            y = dev_ali[utt_id]

            x = model_sa(x)
            pred = model_si(x)

            loss = nllloss(pred, y)
            err = calc_err(pred, y)

            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", 1.0 - err)

    dev_loss = lm.get_stat("dev_loss")
    model_sch.step(dev_loss)

    lm.print_stat()

    parm_path = sa_dir+"/parm/"+str(epoch)+".pt"
    torch.save(model_sa.state_dict(), parm_path)
