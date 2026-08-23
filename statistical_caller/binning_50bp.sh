#!/bin/bash

: '
    Generates 50 bp binned genomic signal profiles from a bedgraph input file.

    The script performs the following steps:
    1. Extracts chromosome lengths from the input bedgraph file.
    2. Generates genome-wide 50 bp intervals using bedtools makewindows.
    3. Calculates overlaps between genomic bins and signal intervals.
    4. Aggregates signal contribution proportionally to overlap length.
    5. Sorts and writes the final 50 bp binned signal BED file.
'

INPUT=$1 
NAME=$(basename "$INPUT" | cut -d'.' -f1) 
bedtools makewindows -w 50 -g <(awk '{if($3>m[$1])m[$1]=$3} END{for(c in m)print c"\t"m[c]}' "$INPUT") \
| bedtools intersect -a - -b "$INPUT" -wo \
| awk '{i=$1":"$2"-"$3; s[i]+=$7*$NF/($3-$2); m[i]=$1"\t"$2"\t"$3} END{for(i in s) print m[i]"\t"s[i]}' \
| sort -k1,1V -k2,2n > "${NAME}_50bp_binned.bed"
