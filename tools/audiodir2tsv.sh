#!/bin/bash

# Set the input and output file paths
transcript_file=$1
audio_dir=$2
output_file=$3

# Write the header row to the output file
echo -e "PATH\tDURATION\tTRANSCRIPT" > $output_file

# Loop over each line in the input file
while IFS= read -r line
do
    # Extract the audio filename and transcript from the line
    audio_filename=$(echo $line | cut -d ' ' -f 1)
    transcription=$(echo $line | cut -d ' ' -f 2-)

    # Get the duration of the audio file
    duration=$(soxi -D $audio_dir/$audio_filename.wav)

    # Write the row to the output file
    echo -e "$audio_dir/$audio_filename.wav\t$duration\t$transcription" >> $output_file
done < "$transcript_file"