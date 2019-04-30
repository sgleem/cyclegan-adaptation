#!/usr/bin/env python3
#coding=utf8

"""
To Do)
    * Split kaldi command factory

Input)

Output)

Requirements)

Role)

Process)
"""
import numpy as np
import kaldi_io
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

    def init_command_book(self):
        """ need to fix
        store for kaldi command
        """
        self.cmd_book["copy-feats"] = kc.copy_feats
        self.cmd_book["apply-cmvn"] = kc.apply_cmvn
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

    def read_to_vec(self):
        print("run",self.cmd)
        generator = kaldi_io.read_vec_int_ark(self.cmd)
        result = {utt_id: np.array(vec) for utt_id, vec in generator}
        return result

def read_feat(feat_path, cmvn_path="", u2s_path="", delta=True):
    km = KaldiReadManager()
    km.set_command("copy-feats", feat_path)
    if cmvn_path is not "" and u2s_path is not "":
        km.set_command("apply-cmvn", u2s_path, cmvn_path)
    if delta:
        km.set_command("add-deltas")
    feat_dict = km.read_to_mat()
    return feat_dict

def read_ali(ali_path, mdl_path):
    km = KaldiReadManager()
    km.set_command("gunzip", ali_path)
    km.set_command("ali-to-pdf", mdl_path)
    ali_dict = km.read_to_vec()
    return ali_dict
