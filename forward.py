import os
import argparse
import torch
import pickle as pk
import numpy as np

from net import *
from module import *
from preprocess import matrix_normalize
from tool.kaldi import kaldi_io as kio
from tool.kaldi.kaldi_manager import read_feat
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/ffmtimit/test", type=str)
parser.add_argument("--si_dir", default="model/gru_si", type=str)
parser.add_argument("--sa_dir", default="", type=str)
args = parser.parse_args()
#####################################################################
data_dir = args.data_dir
si_dir = args.si_dir
sa_dir = args.sa_dir
#####################################################################

os.system("mkdir -p " + si_dir + "/decode")

# data
test_feat = read_feat(data_dir+"/feats.ark", delta=True)

# prior
with open(si_dir+"/prior.pk", 'rb') as f:
    prior = pk.load(f)
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

kaldi_fp = kio.open_or_fd(si_dir+"/decode/lld.ark", 'wb')
with torch.no_grad():
    for utt_id, feat_mat in test_feat.items():
        feat_mat = matrix_normalize(feat_mat, axis=1, fcn_type="mean")
        x = torch.Tensor(feat_mat).cuda().float()
        if sa_dir is not "":
            x = model_sa(x)
        y = dnn_am(x)

        post_prob = y.cpu().numpy()
        lld = post_prob - prior
        kio.write_mat(kaldi_fp, lld, utt_id)
kaldi_fp.close()

