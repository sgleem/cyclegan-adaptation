import os
import argparse
import random

import numpy as np
import pickle as pk
import torch
import torch.optim as opt
import torch.optim.lr_scheduler as sch

import net
import data_preprocess as pp
from tool.log import LogManager
from tool.loss import nllloss, calc_err
from tool.kaldi.kaldi_manager import read_feat, read_ali, read_vec

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--train_feat_dir", default="data/timit/train", type=str)
parser.add_argument("--train_ali_dir", default="ali/timit/train", type=str)
parser.add_argument("--dev_feat_dir", default="data/timit/dev", type=str)
parser.add_argument("--dev_ali_dir", default="ali/timit/dev", type=str)
parser.add_argument("--model_dir", default="model/gru_sat", type=str)
args = parser.parse_args()
#####################################################################
train_feat_dir = args.train_feat_dir
train_ali_dir = args.train_ali_dir
dev_feat_dir = args.dev_feat_dir
dev_ali_dir = args.dev_ali_dir
model_dir = args.model_dir
#####################################################################
epochs = 24
lr = 0.0001
pdf_num = 1920
#####################################################################
os.system("mkdir -p " + model_dir + "/parm")
os.system("mkdir -p " + model_dir + "/opt")

train_feat = read_feat(train_feat_dir+"/feats.ark", cmvn=True, delta=True)
dev_feat = read_feat(dev_feat_dir+"/feats.ark", cmvn=True, delta=True)

train_ali = read_ali(train_ali_dir+"/ali.*.gz", train_ali_dir+"/final.mdl")
dev_ali = read_ali(dev_ali_dir+"/ali.*.gz", dev_ali_dir+"/final.mdl")

train_ivecs = read_vec(train_feat_dir+"/ivectors.ark")
dev_ivecs = read_vec(dev_feat_dir+"/ivectors.ark")

train_utt = list(train_feat.keys())
dev_utt = list(dev_feat.keys())

model = net.GRU_HMM(input_dim=120 * 2, hidden_dim=512, num_layers=5, output_dim=pdf_num)
torch.save(model, model_dir+"/init.pt")
model.cuda()
model_opt = opt.Adam(model.parameters(), lr=lr)
model_sch = sch.ReduceLROnPlateau(model_opt, factor=0.5, patience=0, verbose=True)

model_ivec = net.ivecNN(ivec_dim=100, hidden_dim=512, out_dim=120)
torch.save(model_ivec, model_dir+"/init_ivecNN.pt")
model_ivec.cuda()
model_ivec_opt = opt.Adam(model_ivec.parameters(), lr=lr)
model_ivec_sch = sch.ReduceLROnPlateau(model_ivec_opt, factor=0.5, patience=0, verbose=True)

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
# model training
for epoch in range(epochs):
    print("Epoch",epoch)
    random.shuffle(train_utt)
    model.train()
    lm.init_stat()
    for utt_id in train_utt:
        spk_id = utt_id.split("_")[0]

        x = train_feat[utt_id]
        y = train_ali.get(utt_id,[])
        ivec = train_ivecs[spk_id]
        if len(y) == 0:
            continue

        x = torch.Tensor(x).cuda().float()
        y = torch.Tensor(y).cuda().long()
        ivec = torch.Tensor([ivec]).cuda().float()

        # ivec concat
        aux = model_ivec(ivec)
        aux = aux.expand_as(x)
        x = torch.cat((x,aux), dim=1)

        pred = model(x)
        loss = nllloss(pred, y)
        err = calc_err(pred, y)

        for cur_opt in [model_opt, model_ivec_opt]:
            cur_opt.zero_grad()
        loss.backward()
        for cur_opt in [model_opt, model_ivec_opt]:
            cur_opt.step()

        lm.add_torch_stat("train_loss", loss)
        lm.add_torch_stat("train_acc", 1.0 - err)

    model.eval()
    with torch.no_grad():
        for utt_id in dev_utt:
            spk_id = utt_id.split("_")[0]

            x = dev_feat[utt_id]
            y = dev_ali[utt_id]
            ivec = dev_ivecs[spk_id]

            x = torch.Tensor(x).cuda().float()
            y = torch.Tensor(y).cuda().long()
            ivec = torch.Tensor([ivec]).cuda().float()

            # ivec concat
            aux = model_ivec(ivec)
            aux = aux.expand_as(x)
            x = torch.cat((x,aux), dim=1)

            pred = model(x)
            loss = nllloss(pred, y)
            err = calc_err(pred, y)

            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", 1.0 - err)
    dev_loss = lm.get_stat("dev_loss")
    model_sch.step(dev_loss)
    model_ivec_sch.step(dev_loss)

    lm.print_stat()

    parm_path = model_dir+"/parm/"+str(epoch)+"_si.pt"
    torch.save(model.state_dict(), parm_path)

    parm_path = model_dir+"/parm/"+str(epoch)+"_ivec.pt"
    torch.save(model_ivec.state_dict(), parm_path)

    opt_path = model_dir+"/opt/"+str(epoch)+".pt"
    torch.save(model_opt.state_dict(), opt_path)

