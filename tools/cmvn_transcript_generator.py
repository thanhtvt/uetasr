import argparse
import csv
import random
import os
from itertools import chain
from typing import List, Tuple, Union


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training materials from .tsv files",
    )
    parser.add_argument(
        "--list_tsv",
        nargs="+",
        type=str,
        help="List of .tsv files",
        required=True,
    )
    parser.add_argument(
        "--num_files",
        help="Number of files to generate (if < 1, take percentage)",
        type=float,
        default=1000,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
        default=".",
    )
    parser.add_argument(
        "--outfile_postfix",
        type=str,
        help="Postfix of output files",
        default="",
    )
    parser.add_argument(
        "--full_transcript",
        action="store_true",
        help="Use full transcript for vocabulary creation",
    )
    args = parser.parse_args()
    return args


def get_data_from_files(list_tsv: List[str]):
    data = []
    for tsv in list_tsv:
        tsv_data = []
        with open(tsv, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            reader = list(reader)[1:]   # remove header
            for row in reader:
                tsv_data.append(row)

        data.append(tsv_data)

    return data


def select_data(data: List[Tuple[str, float, str]],
                num_selected: Union[float, int]):
    num_tsv_files = len(data)
    if num_selected > 1:
        num_selected = int(num_selected // num_tsv_files)

    selected_data = []
    for tsv_data in data:
        if num_selected == -1:
            num_selected = len(tsv_data)
        elif num_selected < 1:    # case of percentage
            num_selected = int(len(tsv_data) * num_selected)
        random.shuffle(tsv_data)
        selected_data.extend(tsv_data[:num_selected])

    return selected_data


def generate_cmvn_file(selected_data: List[Tuple[str, float, str]],
                       outfile: str):
    cmvns = []
    for path, *_ in selected_data:
        cmvns.append(path)

    with open(outfile, "w") as f:
        f.write("\n".join(cmvns))


def generate_transcript_text(selected_data: List[Tuple[str, float, str]],
                             outfile: str):
    transcripts = []
    for *_, transcript in selected_data:
        transcripts.append(transcript)

    with open(outfile, "w") as f:
        f.write("\n".join(transcripts))


def main():
    args = parse_args()
    data = get_data_from_files(args.list_tsv)
    selected_data = select_data(data, args.num_files)

    cmvn_outfile = os.path.join(args.output_dir, "cmvn" + args.outfile_postfix + ".tsv")
    generate_cmvn_file(selected_data, cmvn_outfile)

    transcript_outfile = os.path.join(args.output_dir, "transcript" + args.outfile_postfix + ".txt")
    if args.full_transcript:
        generate_transcript_text(chain(*data), transcript_outfile)
    else:
        generate_transcript_text(selected_data, transcript_outfile)


if __name__ == "__main__":
    main()
