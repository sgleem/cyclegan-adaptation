#!/usr/bin/env python3
#coding=utf8

from . import kaldi_io as kio
import numpy as np

def get_ext(file_path):
    fp_component = file_path.split(".")
    if len(fp_component) <= 1:
        return ""
    else:
        return fp_component[-1]

def get_kaldi_mat_generator(file_path):
    """
    return ark or scp generator
    """
    fp = kio.open_or_fd(file_path)
    ext = get_ext(file_path)
    if ext=="ark":
        fp_gen = kio.read_mat_ark(fp)
    elif ext=="scp":
        fp_gen = kio.read_mat_scp(fp)
    else:
        raise NotImplementedError
    return fp_gen

def get_kaldi_mat(file_path, is_np=True):
    """
    return {utt_id:frame_mat numpy or list} dictionary
    """
    mat_dict = dict()
    mat_gen = get_kaldi_mat_generator(file_path)
    for utt_id, frame_mat in mat_gen:
        if is_np:
            mat_dict[utt_id] = np.array(frame_mat)
        else:
            mat_dict[utt_id] = frame_mat
    return mat_dict
