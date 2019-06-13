import os
import argparse
from random import shuffle

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

import net
from tool.log import LogManager
from tool.loss import l2loss
from tool.kaldi.kaldi_manager import read_feat
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--train_source_dir", default="data/ntimit_10%drop/train", type=str)
parser.add_argument("--train_target_dir", default="data/ntimit/train", type=str)
parser.add_argument("--dev_source_dir", default="data/ntimit_10%drop/dev", type=str)
parser.add_argument("--dev_target_dir", default="data/ntimit/dev", type=str)
parser.add_argument("--model_dir", default="model/s2s", type=str)
args = parser.parse_args()
#####################################################################
train_source_dir = args.train_source_dir
train_target_dir = args.train_target_dir
dev_source_dir = args.dev_source_dir
dev_target_dir = args.dev_target_dir
model_dir = args.model_dir
#####################################################################
epochs = 1000
batch_size = 64
lr_e = 0.0001
lr_d = 0.0001

recon_coef = 10.0; prob_coef = 1.0
#####################################################################
torch.cuda.empty_cache()
os.system("mkdir -p "+ model_dir +"/parm")

train_source = read_feat(train_source_dir+"/feats.ark", utt_cmvn=True, delta=True)
train_target = read_feat(train_target_dir+"/feats.ark", utt_cmvn=True, delta=True)
dev_source = read_feat(dev_source_dir+"/feats.ark", utt_cmvn=True, delta=True)
dev_target = read_feat(dev_target_dir+"/feats.ark", utt_cmvn=True, delta=True)

train_utt = list(train_source.keys())
dev_utt = list(dev_source.keys())

train_utt.sort(key=lambda x: len(train_source[x]))
train_utt.reverse()

encoder = net.SelfAttnEncoder(input_size=120, mha_size=120, ffn_size=512, enc_num=1, n_head=4)
decoder = net.SelfAttnDecoder(input_size=120, mha_size=120, ffn_size=512, dec_num=1, n_head=4)

enc_opt = optim.Adam(encoder.parameters(), lr=lr_e)
dec_opt = optim.Adam(decoder.parameters(), lr=lr_d)

enc_sch = sched.StepLR(enc_opt, step_size=100, gamma=0.5)
dec_sch = sched.StepLR(dec_opt, step_size=100, gamma=0.5)

lm = LogManager()
lm.alloc_stat_type_list(["train_recon", "train_end", "dev_recon", "dev_end"])

for model in [encoder, decoder]:
    model.cuda()

sos = torch.zeros(1, 120).cuda().float()
eos = torch.zeros(1, 120).cuda().float()

torch.save(encoder, model_dir+"/encoder.pt")
torch.save(decoder, model_dir+"/decoder.pt")

for epoch in range(epochs):
    print("EPOCH :", epoch)
    lm.init_stat()

    # shuffle(train_utt)
    for model in [encoder, decoder]:
        model.train()
    for utt_id in train_utt:
        source = train_source[utt_id] # (T, 120)
        target = train_target[utt_id] # (T, 120)

        source = torch.Tensor(source).cuda().float()
        target = torch.Tensor(target).cuda().float()

        source_enc = encoder(source) # (T, 120)
        
        pred, end_prob = decoder(enc=source_enc, prev=sos) # (1, 120) (1, 1)
        recon = 0; prob = 0
        for teacher in target:
            # (120) -> (1, 120)
            recon += l2loss(pred, teacher) / 2
            prob += l2loss(end_prob, 0) / 2
            pred, end_prob = decoder(enc=source_enc, prev=teacher.unsqueeze(dim=0))
        recon += l2loss(pred, eos) / 2
        prob += l2loss(end_prob, 1) / 2
        
        recon = torch.mean(recon)
        prob = torch.mean(prob)

        total = (recon_coef * recon) + (prob_coef * prob)

        for opt in [enc_opt, dec_opt]:
            opt.zero_grad()
        total.backward()
        for opt in [enc_opt, dec_opt]:
            opt.step()

        lm.add_torch_stat("train_recon", recon)
        lm.add_torch_stat("train_end", prob)

    for model in [encoder, decoder]:
        model.eval()
    with torch.no_grad():
        for utt_id in dev_utt:
            source = dev_source[utt_id] # (T, 120)
            target = dev_target[utt_id] # (T, 120)

            source = torch.Tensor(source).cuda().float()
            target = torch.Tensor(target).cuda().float()

            source_enc = encoder(source) # (T, 120)
            
            pred, end_prob = decoder(enc=source_enc, prev=sos) # (1, 120) (1, 1)
            recon = 0; prob = 0
            for teacher in target:
                # (120) -> (1, 120)
                recon += l2loss(pred, teacher) / 2
                prob += l2loss(end_prob, 0) / 2
                pred, end_prob = decoder(enc=source_enc, prev=teacher.unsqueeze(dim=0))
            recon += l2loss(pred, eos) / 2
            prob += l2loss(end_prob, 1) / 2
            
            recon = torch.mean(recon)
            prob = torch.mean(prob)

            lm.add_torch_stat("dev_recon", recon)
            lm.add_torch_stat("dev_end", prob)

    lm.print_stat()

    parm_path = model_dir+"/parm/"+str(epoch)+"_enc.pt"
    torch.save(encoder.state_dict(), parm_path)

    parm_path = model_dir+"/parm/"+str(epoch)+"_dec.pt"
    torch.save(decoder.state_dict(), parm_path)