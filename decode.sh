#!/bin/bash

data_dir=$1
model_dir=$2

if [ $# -eq 3 ]; then
    gen_dir=$3
    python3 forward.py --data_dir $data_dir --si_dir $model_dir --sa_dir $gen_dir
else
    python3 forward.py --data_dir $data_dir --si_dir $model_dir
fi

. cmd.sh
lld="ark,s,cs:cat $model_dir/decode/lld.ark |"
latgen-faster-mapped --min-active=200 --max-active=7000 \
    --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 \
    --acoustic-scale=0.2 --allow-partial=true \
    gmm_hmm/final.mdl gmm_hmm/graph/HCLG.fst "$lld" \
    "ark:|gzip -c > $model_dir/decode/lat.1.gz"

bash tool/score/timit/score.sh --cmd "$decode_cmd" \
    $data_dir gmm_hmm/graph $model_dir/decode

bash tool/score/result.sh $model_dir