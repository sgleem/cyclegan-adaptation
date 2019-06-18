import os
import sys
import argparse
import torch
import pickle as pk
import numpy as np

from net import *
from module import *
import data_preprocess as pp
from tool.kaldi import kaldi_io as kio
from tool.kaldi.kaldi_manager import read_feat, read_vec
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/timit/test", type=str)
parser.add_argument("--si_dir", default="model/gru_wsj", type=str)
parser.add_argument("--sa_dir", default="", type=str)
args = parser.parse_args()
#####################################################################
data_dir = args.data_dir
si_dir = args.si_dir
sa_dir = args.sa_dir
#####################################################################
loss_per = 0

os.system("mkdir -p " + si_dir + "/decode")
# data
test_feat = read_feat(data_dir+"/feats.ark", utt_cmvn=True, delta=True)
test_utts = list(test_feat.keys())
test_utts.sort()

# prior
with open(si_dir+"/prior.pk", 'rb') as f:
    prior = pk.load(f)
    print(len(prior))
    prior = np.log(prior).reshape(-1, len(prior))

# dnn
dnn_am = torch.load(si_dir+"/init.pt")
dnn_am.load_state_dict(torch.load(si_dir+"/final.pt"))
dnn_am.cuda()
dnn_am.eval()
if sa_dir is not "":
    model_sa = torch.load(sa_dir+"/init.pt")
    model_sa.load_state_dict(torch.load(sa_dir+"/final.pt"))
    model_sa.cuda()
    model_sa.eval()
    print(model_sa)
# if sa_dir is not "":
#     G = torch.load(sa_dir+"/init_G.pt")
#     G.load_state_dict(torch.load(sa_dir+"/final_G.pt"))
#     G.cuda()
#     G.eval()

#     P = torch.load(sa_dir+"/init_packet.pt")
#     P.load_state_dict(torch.load(sa_dir+"/final_packet.pt"))
#     P.cuda()
#     P.eval()

kaldi_fp = kio.open_or_fd(si_dir+"/decode/lld.ark", 'wb')
with torch.no_grad():
    for utt_id in test_utts:
        feat_mat = test_feat[utt_id]
        if sa_dir is not "":
            x = torch.Tensor([feat_mat]).cuda().float()
            x = model_sa(x)[0]
        # if sa_dir is not "":
        #     x = torch.Tensor([feat_mat]).cuda().float()
        #     prob = P(x)[0]
        #     recon = G(x)[0]
        #     x = prob * x[0] + (1-prob) * recon

        else:
            x = torch.Tensor(feat_mat).cuda().float()
        y = dnn_am(x)

        post_prob = y.cpu().numpy()
        lld = post_prob - prior
        kio.write_mat(kaldi_fp, lld, utt_id)
kaldi_fp.close()

