#!/bin/bash

perl -e 'while(<>){ 
    s/\|(\d)/\| $1/g; s/(\d)\|/$1 \|/g;
    if (m/[WS]ER (\S+)/ && (!defined $bestwer || $bestwer > $1)){ $bestwer = $1; $bestline=$_; } # kaldi "compute-wer" tool.
    elsif (m: (Mean|Sum/Avg|)\s*\|\s*\S+\s+\S+\s+\|\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s+\|:
        && (!defined $bestwer || $bestwer > $2)){ $bestwer = $2; $bestline=$_; } }  # sclite.
   if (defined $bestline){ print $bestline; } ' | \
  awk 'BEGIN{ FS="%WER"; } { if(NF == 2) { print FS$2" "$1; } else { print $0; }}' | \
  awk 'BEGIN{ FS="Sum/Avg"; } { if(NF == 2) { print $2" "$1; } else { print $0; }}' | \
  awk '{ if($1!~/%WER/) { print "%WER "$9" "$0; } else { print $0; }}' | \
  sed -e 's|\s\s*| |g' -e 's|\:$||' -e 's|\:\s*\|\s*$||'