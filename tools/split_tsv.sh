#!/bin/bash

# Get the input TSV file, validation set ratio (as a decimal), shuffle option, and output filename postfix (optional) from the command line arguments
TSV_FILE=$1
VAL_RATIO=$2
SHUFFLE=$3
POSTFIX=${4:-""}

# Define the output TSV files
TRAIN_FILE="${TSV_FILE%.*}_train$POSTFIX.tsv"
VALID_FILE="${TSV_FILE%.*}_val$POSTFIX.tsv"

# Write the TSV headers to the output files
echo -e "PATH\tDURATION\tTRANSCRIPT" > $TRAIN_FILE
echo -e "PATH\tDURATION\tTRANSCRIPT" > $VALID_FILE

# Initialize the total duration of the validation set to 0
TOTAL_DURATION=0
VALID_DURATION=0

# Read the lines from the input TSV file into an array
LINES=()
while read -r LINE; do
    LINES+=("$LINE")
    # Get the duration from the DURATION column and add it to the total duration
    DURATION=$(echo "$LINE" | cut -f 2)
    TOTAL_DURATION=$(echo "$TOTAL_DURATION + $DURATION" | bc)
done < $TSV_FILE

# Shuffle the lines if requested
if [ "$SHUFFLE" = "shuffle" ]; then
    shuf -e "${LINES[@]}" > shuffled.tsv
    LINES=()
    while read -r LINE; do
        LINES+=("$LINE")
    done < shuffled.tsv
fi

# Calculate the total duration of the validation set
VALID_DURATION=$(echo "$TOTAL_DURATION * $VAL_RATIO" | bc)
TRAIN_DURATION=$(echo "$TOTAL_DURATION - $VALID_DURATION" | bc)

# Loop over each line in the input TSV file
for LINE in "${LINES[@]}"; do
    # Get the duration from the DURATION column
    DURATION=$(echo "$LINE" | cut -f 2)
    
    # If adding this duration to the validation set would exceed
    # the total duration, write the line to the train file instead
    if (( $(echo "$VALID_DURATION < $DURATION" | bc -l) )); then
        echo "$LINE" >> $TRAIN_FILE
        TRAIN_DURATION=$(echo "$TRAIN_DURATION - $DURATION" | bc)
    else
        echo "$LINE" >> $VALID_FILE
        VALID_DURATION=$(echo "$VALID_DURATION - $DURATION" | bc)
    fi
done

# Remove the shuffled file if it was created
if [ "$SHUFFLE" = "true" ]; then
    rm shuffled.tsv
fi