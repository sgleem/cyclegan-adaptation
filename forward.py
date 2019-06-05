import os
import argparse
import torch
import pickle as pk
import numpy as np

from net import *
from module import *
from data_preprocess import matrix_normalize
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

os.system("mkdir -p " + si_dir + "/decode")

# data
test_feat = read_feat(data_dir+"/feats.ark", cmvn=True, delta=True)
# ivector
test_ivecs = read_vec(data_dir+"/ivectors.ark")

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

# for sat
ivecnn = torch.load(si_dir+"/init_ivecNN.pt")
ivecnn.load_state_dict(torch.load(si_dir+"/final_ivecNN.pt"))
ivecnn.cuda()
ivecnn.eval()

if sa_dir is not "":
    model_sa = torch.load(sa_dir+"/init.pt")
    model_sa.load_state_dict(torch.load(sa_dir+"/final.pt"))
    model_sa.cuda()
    model_sa.eval()
    print(model_sa)

kaldi_fp = kio.open_or_fd(si_dir+"/decode/lld.ark", 'wb')
with torch.no_grad():
    for utt_id, feat_mat in test_feat.items():
        
        if sa_dir is not "":
            x = torch.Tensor([feat_mat]).cuda().float()
            x = model_sa(x)[0]
        else:
            x = torch.Tensor(feat_mat).cuda().float()
        # for sat
        spk_id = utt_id.split("_")[0]
        ivec = test_ivecs[spk_id]
        ivec = torch.Tensor([ivec]).cuda().float()
        aux = ivecnn(ivec)
        aux = aux.expand_as(x)
        x = torch.cat((x,aux), dim=1)
        # model_si
        y = dnn_am(x)

        post_prob = y.cpu().numpy()
        lld = post_prob - prior
        kio.write_mat(kaldi_fp, lld, utt_id)
kaldi_fp.close()

