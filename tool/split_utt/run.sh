#!/bin/bash

# Example)
#   script/split_file_storage/run.sh data/noise/test.fbk data/adaptation
# In Prepare)
# Input: 1. $original_data_dir/spk2utt
#       2. $data_root_for_adaptaion
# ex) original_data_dir = data/test
#    data_root_for_adaptation = data/adaptation
# Output: 1. $data_root_for_adaptation/{adapt, dev, test}/uttlist
#
# In Run)
# Input: 1. $data_root_for_adaptation/{adapt, dev, test}/uttlist
#        2. $original_data_dir/
#               {glm, stm, text, feats.scp, spk2utt, utt2spk, wav.scp, cmvn.scp}
# ex) data_root_for_adaptation = data/adaptation
#     original_data_dir = data/test
# Output: $data_root_for_adaptation/{adapt, dev, test}/
#          {glm, stm, text, feats.scp, spk2utt, utt2spk, wav.scp, cmvn.scp}
#
# Requirements)
# 1. utt2spk_to_spk2utt.pl
# 2. compute_cmvn_stats.sh
# 3. validate_data_dir.sh
#   1) validate_text.pl
#   2) spk2utt_to_utt2spk.pl
#
# Process)
# 0. run prepare.py
# For each [adapt, dev, test]
#   
#   1. Read Uttlist ( cat | sort | uniq )
#   2. cp [glm]: if exist, copy
#   3. reset other file if exist [stm, text, feats.scp, utt2spk, wav.scp]
#   4. preprocess stm(first 3 line)    
#   5. find first field [stm, text, feats.scp, utt2spk, wav.scp]
#   6. get spk2utt by executing script
#   7. cmvn compute
#   8. data validation
#
# each cmvn is computed by small feat set, which makes cmvn stats inaccurate
# In adaptation, cmvn is not recommended
# For this reason, cmvn step is erased.
if [ $# -lt 5 ]; then
    echo "Wrong command"
    echo "bash run.sh data/corpus data/adaptation 4 2 2"
fi
original_data_dir=$1
data_root_for_adaptation=$2
train_num=$3
dev_num=$4
test_num=$5
start_idx=0
if [ $# -gt 5 ]; then
    start_idx=$6
fi


# 0. run prepare.py
package_dir=`dirname $0`
python3 ${package_dir}/prepare.py $original_data_dir $data_root_for_adaptation $train_num $dev_num $test_num $start_idx  || exit 1;


inDir=$original_data_dir
for dType in adapt dev test; do
    outDir=${data_root_for_adaptation}/${dType}

    #   1. Read Uttlist ( cat | sort | uniq )
    uttList=( $( cat ${outDir}/uttlist ) )

    #   2. cp [glm, spk2gender]: if exist, copy
    for file in glm spk2gender; do
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
    rm $outDir/feats.scp
    mv $outDir/feats.ark $outDir/feats.scp
done