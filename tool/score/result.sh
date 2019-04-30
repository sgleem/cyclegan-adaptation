#!/bin/bash

check_dir=./model/gru*
if [ $# -gt 1 ]; then
    check_dir=$1
fi

score=`dirname $0`
for x in $check_dir/decode*; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | bash $score/wer.sh; done
for x in $check_dir/decode*; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/score_*/*.sys 2>/dev/null | bash $score/wer.sh; done