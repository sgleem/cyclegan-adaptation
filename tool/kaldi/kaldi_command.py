#!/usr/bin/env python3
#coding=utf8
def get_specifier(file_name, option=[]):
    ext = file_name.split(".")[-1]
    if ext not in ["scp", "ark"]:
        ext = "ark"
    specifier =  ext + ":" + file_name
    return specifier

def make_kaldi_cmd(kaldi_cmd, *args):
    result_cmd = kaldi_cmd +" " + " ".join(args) + " |"
    return result_cmd

def copy_feats(feat_path="-", out_path="-"):
    """
    copy-feats <ark,scp:feats.ark,scp> <ark:-> |
    """
    kaldi_cmd="copy-feats"
    r_specifier=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, r_specifier, o_specifier)
    return result_cmd

def apply_cmvn(u2s_path, cmvn_path, feat_path="-", out_path="-"):
    """
    apply-cmvn <--utt2spk=ark:utt2spk> <scp:cmvn.scp> <ark:feats.scp> <ark:-> |
    """
    kaldi_cmd="apply-cmvn"
    u2s="--utt2spk="+get_specifier(u2s_path)
    r_specifier1=get_specifier(cmvn_path)
    r_specifier2=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, u2s, r_specifier1, r_specifier2, o_specifier)
    return result_cmd
def add_deltas(feat_path="-", out_path="-"):
    """
    add-deltas <scp:feats.scp> <ark:-> |
    """
    kaldi_cmd="add-deltas"
    r_specifier=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, r_specifier, o_specifier)
    return result_cmd
def gunzip(ali_path):
    """
    gunzip -c <ali.1.gz> |
    """
    kaldi_cmd = "gunzip"
    option="-c"
    result_cmd = make_kaldi_cmd(kaldi_cmd,option,ali_path)
    return result_cmd
def ali_to_pdf(mdl_path, ali_path="-", out_path="-"):
    """
    ali-to-pdf <final.mdl> <ark:ali.1.ark> <ark:-> |
    """
    kaldi_cmd="ali-to-pdf"
    r_specifier = get_specifier(ali_path)
    o_specifier = get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, mdl_path, r_specifier, o_specifier)
    return result_cmd
def splice_feats(left, right, feat_path="-", out_path="-"):
    """
    splice-feats <--left-context=5> <--right-context=5> <ark:feats.ark> <ark:-> |
    """
    kaldi_cmd="splice-feats"
    option1="--left-context="+left
    option2="--right-context="+right
    r_specifier=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    result_cmd = make_kaldi_cmd(kaldi_cmd, option1, option2, r_specifier, o_specifier)
    return result_cmd
