#!/usr/bin/env python3
#coding=utf8
import os
import numpy as np
from . import kaldi_io
from . import kaldi_command as kc

class KaldiReadManager:
    """
    Read kaldi data from HDD by using assemblized kaldi command
    Note that read method is generator
    """
    def __init__(self):
        self.cmd = ""
        self.cmd_book = dict()
        self.init_command()
        self.init_command_book()
    
    def run(self):
        self.cmd = self.cmd[:-1] # delete |
        os.system(self.cmd)

    def init_command_book(self):
        """ need to fix
        store for kaldi command
        """
        self.cmd_book["copy-feats"] = kc.copy_feats
        self.cmd_book["copy-vector"] = kc.copy_vector
        self.cmd_book["apply-cmvn"] = kc.apply_cmvn
        self.cmd_book["compute-cmvn-stats"] = kc.compute_cmvn_stats
        self.cmd_book["add-deltas"] = kc.add_deltas
        self.cmd_book["splice-feats"] = kc.splice_feats
        self.cmd_book["gunzip"] = kc.gunzip
        self.cmd_book["ali-to-pdf"] = kc.ali_to_pdf

    def init_command(self):
        self.cmd = ""

    def set_command(self, command, *args, **kwargs):
        assert command in self.cmd_book, "wrong kaldi command"
        cur_command = self.cmd_book[command](*args, **kwargs)
        self.cmd += cur_command

    def read_to_mat(self):
        print("run",self.cmd)
        generator = kaldi_io.read_mat_ark(self.cmd)
        result = {utt_id: np.array(frame_mat) for utt_id, frame_mat in generator}
        return result

    def read_to_vec(self, type='int'):
        print("run",self.cmd)
        if type=='int':
            generator = kaldi_io.read_vec_int_ark(self.cmd)
        if type=='float':
            generator = kaldi_io.read_vec_flt_ark(self.cmd)
        result = {utt_id: np.array(vec) for utt_id, vec in generator}
        return result

def read_feat(feat_path, cmvn=True, delta=True):
    km = KaldiReadManager()
    
    if cmvn:
        km.set_command("compute-cmvn-stats", feat_path=feat_path, out_path="cmvn.ark")
        km.run()
    
    km.init_command()
    km.set_command("copy-feats", feat_path)
    if cmvn:
        km.set_command("apply-cmvn", cmvn_path="cmvn.ark")
    if delta:
        km.set_command("add-deltas")
    feat_dict = km.read_to_mat()
    return feat_dict

def read_ali(ali_path, mdl_path):
    km = KaldiReadManager()
    km.set_command("gunzip", ali_path)
    km.set_command("ali-to-pdf", mdl_path)
    ali_dict = km.read_to_vec(type='int')
    return ali_dict

def read_vec(vec_path):
    km = KaldiReadManager()
    km.set_command("copy-vector", vec_path)
    ali_dict = km.read_to_vec(type='float')
    return ali_dict
