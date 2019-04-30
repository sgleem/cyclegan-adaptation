#!/usr/bin/env python3
#coding=utf8

"""
Input:  1. $original_data_dir/spk2utt
        2. $data_root_for_adaptaion
        3. $envPath
ex) original_data_dir = data/test
   data_root_for_adaptation = data/adaptation

Output: 1. $data_root_for_adaptation/{adapt, dev, test}/uttlist

Requirements)
1. split.conf
  - adapt_txt_num: 4
  - dev_txt_num: 2
  - test_txt_num: 2
  - start_idx: 0

Process) 
0. Read configuration
1. Read spk2utt
2. Save into {spkId: txtList}
3. Get [adapt, dev, test] uttList set
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
4. sort each set
5. make {adapt, dev, test} dir
6. save to {adapt, dev, test}/uttlist
"""
import os
import sys
import json
    
# 2. Save into {spkId: txtList}
def readS2U(s2uPath, token=" "):
    s2u = dict()
    with open(s2uPath, 'r') as f:
        for line in f:
            line = line[:-1]
            spkId = line.split(token)[0]
            uttList = line.split(token)[1:]

            s2u[spkId] = uttList
    return s2u

# 6. save to {adapt, dev, test}/uttlist
def saveUttList(uttList, setDir):
    # 4. sort each set
    uttList.sort()
    # 6. save to {adapt, dev, test}/uttlist
    uttListPath = setDir+"/uttlist"
    with open(uttListPath, 'w') as f:
        for uttId in uttList:
            f.write(uttId+"\n")

# 3. Get [adapt, dev, test] uttList set
# dataSet = {"adapt":[],"dev":[],"test":[]}
def extractSet(s2u, adapt_num, dev_num, test_num, start_idx=0):
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
    total_num = adapt_num + dev_num + test_num
    num = {"adapt" : adapt_num, "dev" : dev_num, "test" : test_num}
    dataSet = {"adapt" : [], "dev" : [], "test": []}

    for uttList in s2u.values():
        # 1, 2, 3) Iterate Set
        for setType in ["adapt", "dev", "test"]:
            set_num = num[setType]
            end_idx = (start_idx + set_num)
            for cur_idx in range(start_idx, end_idx):
                cur_idx = cur_idx % total_num
                cur_uttId = uttList[cur_idx]
                dataSet[setType].append(cur_uttId)
            start_idx = end_idx
    
    return dataSet





if __name__=="__main__":
    inDir = sys.argv[1] # original_data_dir
    outDir = sys.argv[2] # data_root_for_adaptation
    adapt_num = int(sys.argv[3])
    dev_num = int(sys.argv[4])
    test_num = int(sys.argv[5]) 
    start_idx = int(sys.argv[6]) 

    # 1. Read spk2utt
    # 2. Save into {spkId: txtList}
    s2uPath=inDir+"/spk2utt"
    s2u = readS2U(s2uPath)

    # 3. Get [adapt, dev, test] uttList set
    # dataSet = {"adapt":[],"dev":[],"test":[]}
    dataSet = extractSet(s2u, adapt_num, dev_num, test_num, start_idx)

    # 4. sort each set
    for eachSet in dataSet.values():
        eachSet.sort()

    # 5. make {adapt, dev, test} dir
    # 6. save to {adapt, dev, test}/uttlist
    for setType, uttList in dataSet.items():
        setDir = outDir+"/"+setType
        # 5. make {adapt, dev, test} dir
        os.system("mkdir -p " + setDir)
        # 6. save to {adapt, dev, test}/uttlist
        saveUttList(uttList, setDir)