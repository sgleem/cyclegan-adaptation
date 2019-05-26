#!/bin/bash
master=speech102
clusters=("speech102" "speech103" "speech104" "speech105")
size=${#clusters[@]}
cur_dir=/nfs/${master}/hdd/lsg9311/cyclegan-adaptation
rm -rf $cur_dir/sharedfile # for sharedfile clean

for (( i=0 ; i < $size ; i++ )) ; do
    node=${clusters[$i]}
    echo $node execute
    ssh $node "nohup python3 -u $cur_dir/train_dist_si.py -r $i -s $size -m $master > $cur_dir/log/${node}_wsj_si.txt" &
    # ssh $node "python3 -u $cur_dir/train_dist_si.py -r $i -s $size -m $master" &
done

rm -rf $cur_dir/sharedfile # for sharedfile clean