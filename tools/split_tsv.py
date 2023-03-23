import argparse
import csv
import random
import os


def split_dataset(input_file, val_ratio, shuffle=False, postfix=""):
    # Define the chunk size for reading the input file
    chunk_size = 5000

    # Open the input TSV file for reading
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')

        # Determine the total duration of the dataset
        total_duration = 0
        for row in reader:
            if row[1] != "DURATION":
                total_duration += float(row[1])

        # Calculate the duration of the validation set
        val_duration = total_duration * val_ratio
        train_duration = total_duration - val_duration

        # Seek back to the beginning of the file
        f.seek(0)

        # Open the output files for writing
        train_file = os.path.splitext(input_file)[0] + "_train" + postfix + ".tsv"
        val_file = os.path.splitext(input_file)[0] + "_val" + postfix + ".tsv"
        with open(train_file, 'w', newline='') as train_f, open(val_file, 'w', newline='') as val_f:
            writer_train = csv.writer(train_f, delimiter='\t')
            writer_val = csv.writer(val_f, delimiter='\t')
            writer_train.writerow(["PATH", "DURATION", "TRANSCRIPT"])
            writer_val.writerow(["PATH", "DURATION", "TRANSCRIPT"])

            # Loop over the input file in chunks
            while True:
                # Read the next chunk of rows from the input file
                chunk = []
                for i in range(chunk_size):
                    row = next(reader, None)
                    if row is None:
                        break
                    chunk.append(row)

                # Shuffle the chunk if requested
                if shuffle:
                    random.shuffle(chunk)

                # Loop over the rows in the chunk and write them to the output files
                for row in chunk:
                    if row[1] == "DURATION":
                        continue
                    duration = float(row[1])
                    if val_duration < duration:
                        writer_train.writerow(row)
                        train_duration -= duration
                    else:
                        writer_val.writerow(row)
                        val_duration -= duration
                    if val_duration < 0 and abs(val_duration) > train_duration:
                        last_row = row
                        writer_train.writerow(last_row)
                        train_duration -= duration

                # If we've reached the end of the file, break out of the loop
                if len(chunk) < chunk_size:
                    break

    print(f"Train file written to {train_file}")
    print(f"Validation file written to {val_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a TSV dataset into train and validation sets.")

    parser.add_argument(
        "input_file",
        type=str,
        help="input TSV file to split"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="ratio of dataset to use for validation (as a decimal)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle the lines before splitting"
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="postfix to append to output filenames"
    )
    args = parser.parse_args()
    split_dataset(args.input_file, args.val_ratio, args.shuffle, args.postfix)
