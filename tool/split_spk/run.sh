#!/bin/bash

# Example)
#   tool/split_spk/run.sh data/ntimit/train data/ntimit 231
# In Prepare)
# Input: 1. $in_dir/spk2utt
#        2. $out_dir
# Output: 2. $out_dir/{adapt, dev}/uttlist
#
# In Run)
# Input: 1. $out_dir/{adapt, dev}/uttlist
#        2. $in_dir/
#               {glm, stm, text, feats.scp, spk2utt, utt2spk, wav.scp, cmvn.scp}
# Output: $out_dir/{adapt, dev}/
#          {glm, stm, text, feats.scp, spk2utt, utt2spk, wav.scp, cmvn.scp}


if [ $# -lt 3 ]; then
    echo "bash tool/split_spk/run.sh data/ntimit/train data/ntimit 231"
    exit 0;
fi
inDir=$1
rootDir=$2
train_num=$3
start_idx=0
if [ $# -gt 3 ]; then
    start_idx=$4
fi
total_num=462
if [ $# -gt 4 ]; then
    total_num=$5
fi

# 0. run prepare.py
package_dir=`dirname $0`
mkdir -p ${rootDir}/adapt
python3 ${package_dir}/prepare.py $inDir $rootDir/adapt $train_num $total_num $start_idx || exit 1;


for dType in adapt; do
    
    outDir=${rootDir}/${dType}

    #   1. Read Uttlist ( cat | sort | uniq )
    uttList=( $( cat ${outDir}/uttlist ) )

    #   2. cp [glm]: if exist, copy
    for file in glm; do
        [ -f $inDir/$file ] && cp $inDir/$file $outDir/$file
    done
    
    #   3. reset other file if exist [stm, text, feats.scp, utt2spk, wav.scp]
    for file in stm text feats.scp utt2spk wav.scp; do
        [ -f $outDir/$file ] && rm $outDir/$file
        [ -f $inDir/$file ] && touch $outDir/$file
    done

    #   4. preprocess stm(first 3 line)    
    if [ -f $inDir/stm ]; then
        head -n 3 $inDir/stm > $outDir/stm
    fi
    
    #   5. find first field [stm, text, feats.scp, utt2spk, wav.scp]
    for uttId in ${uttList[@]}; do
        for file in stm text feats.scp utt2spk wav.scp; do
            if [ -f $inDir/$file ]; then
                cat $inDir/$file | grep -w $uttId >> $outDir/$file
            fi
        done
    done

    #   6. get spk2utt by executing script
    cat $outDir/utt2spk | ${package_dir}/utt2spk_to_spk2utt.pl > $outDir/spk2utt || exit 1;

    #   8. data validation
    ${package_dir}/validate_data_dir.sh $outDir || exit 1;

    # 9. copy to ark
    copy-feats scp,s:$outDir/feats.scp ark:$outDir/feats.ark
done
