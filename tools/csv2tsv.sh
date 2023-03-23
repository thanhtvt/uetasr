#!/bin/bash

# Set the input and output file paths
input_file=$1
output_file=$2

# Transform the file using awk
awk 'BEGIN { FS=","; OFS="\t" } { print $0 }' $input_file > $output_file