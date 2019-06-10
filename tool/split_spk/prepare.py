#!/usr/bin/env python3
#coding=utf8

"""
"""
import os
import sys
import json

# 1. Read spk2utt
# 2. Save into {spkId: txtList}
def read_s2u(s2uPath, token=" "):
    s2u = dict()
    with open(s2uPath, 'r') as f:
        for line in f:
            line = line[:-1]
            spkId = line.split(token)[0]
            uttList = line.split(token)[1:]

            s2u[spkId] = uttList
    return s2u



# 3. Get [adapt, dev, test] uttList set
# dataSet = {"adapt":[],"dev":[],"test":[]}
def extract_set(s2u, adapt_num, dev_num, start_idx=0):
    """
    for each txtList in spkId
        0) check if split is valid, if not raise error
        1) adapt_start ~ adapt_end: append utt to adapt set
            adapt_start: (start_idx) % total_len
            adapt_end: (start_idx+adapt_txt_num) % total_len
        2) dev_start ~ dev_end: append utt to dev set
            dev_start: (adapt_end) % total_len
            dev_end: (adapt_end+dev_txt_num) % total_len
        3) test_start ~ test_end: append utt to test set
            test_start: (dev_end) % total_len
            test_end: (dev_end+test_txt_num) % total_len
    """
    num = {"adapt": adapt_num}
    data_set = {"adapt": []}
    total_num = adapt_num + dev_num
    spk_list = list(s2u.keys())

    # 1, 2, 3) Iterate Set
    for set_type in ["adapt"]:
        set_num = num[set_type]
        end_idx = (start_idx + set_num)
        for cur_idx in range(start_idx, end_idx):
            cur_idx = cur_idx % total_num
            cur_spk = spk_list[cur_idx]
            cur_utt_list = s2u[cur_spk]
            data_set[set_type].extend(cur_utt_list)
        start_idx = end_idx
    
    return data_set

# 6. save to {adapt, dev, test}/uttlist
def saveUttList(uttList, setDir):
    # 4. sort each set
    uttList.sort()
    # 6. save to {adapt, dev, test}/uttlist
    uttListPath = setDir+"/uttlist"
    with open(uttListPath, 'w') as f:
        for uttId in uttList:
            f.write(uttId+"\n")

if __name__=="__main__":
    in_dir = sys.argv[1] # original_data_dir
    out_dir = sys.argv[2] # adaptation_dir
    adapt_num = int(sys.argv[3])
    total_num = int(sys.argv[4]) 
    start_idx = int(sys.argv[5]) 

    # 1. Read spk2utt
    # 2. Save into {spkId: txtList}
    s2u_path=in_dir+"/spk2utt"
    s2u = read_s2u(s2u_path)

    # 3. Get [adapt, dev] uttList set
    # data_set = {"adapt":[],"dev":[]}
    data_set = extract_set(s2u, adapt_num, total_num-adapt_num, start_idx)

    # 4. sort each set
    for each_set in data_set.values():
        each_set.sort()

    # 5. make {adapt, dev, test} dir
    # 6. save to {adapt, dev, test}/uttlist
    for set_type, uttList in data_set.items():
        set_dir = out_dir
        # 5. make {adapt, dev, test} dir
        os.system("mkdir -p " + set_dir)
        # 6. save to {adapt, dev, test}/uttlist
        saveUttList(uttList, set_dir)
