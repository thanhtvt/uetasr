#!/bin/bash

# Set the input file and the string to be replaced
input_file=$1
output_word=$(echo "$input_file" | sed 's/_/-/g')
replace_string="/home/namnv59/datahub/librispeech/${output_word%.tsv}"
echo "$replace_string"
new_string="/work/hpc/iai/thanhtvt"

# Use awk to replace the string in the first column
awk -v old="$replace_string" -v new="$new_string" 'BEGIN { FS="\t" } { i = index($1, old); if (i > 0) $1 = substr($1, 1, i-1) new substr($1, i+length(old)); print }' "$input_file" > temp.tsv && mv temp.tsv "$input_file"
