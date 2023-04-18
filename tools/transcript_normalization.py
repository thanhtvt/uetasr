import csv
import string
import sys


# read tsv file
def read_csv(filename, delimiter='\t'):
    with open(filename, 'r') as f:
        # read csv file except the first line
        reader = csv.reader(f, delimiter=delimiter)
        reader = list(reader)[1:]
        lines = []
        for line in reader:
            lines.append(line)
    return lines


# write tsv file
def write_csv(filename, lines, delimiter='\t'):
    headers = ["PATH", "DURATION", "TRANSCRIPT"]
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(lines)


class TextNormalization(object):
    def __init__(self, csv_filepath: str):
        self.lines = read_csv(csv_filepath)
        # get only the third column
        self.transcripts = [line[2] for line in self.lines]

    def get_csv_rows(self):
        return self.lines

    def normalize(self):
        normalized_transcripts = []
        for transcript in self.transcripts:
            normalized_transcripts.append(self.text_preprocessing(transcript))
        return normalized_transcripts

    def text_preprocessing(self, text: str):
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numeric(text)
        text = self.normalize_accents(text)
        return text

    @staticmethod
    def lowercase(sentence: str):
        return sentence.lower()

    @staticmethod
    def remove_punctuation(sentence: str):
        return sentence.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_numeric(sentence: str):
        return ' '.join([word for word in sentence.split() if word.isalpha()])

    @staticmethod
    def normalize_accents(sentence: str):
        return replace_all(sentence, dict_map)


# coding=utf-8
# Copyright (c) 2021 VinAI Research

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}


def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


if __name__ == '__main__':
    csv_filepath = sys.argv[1]

    text_normalizer = TextNormalization(csv_filepath)
    normalized_transcripts = text_normalizer.normalize()

    lines = text_normalizer.get_csv_rows()
    for i, line in enumerate(lines):
        line[2] = normalized_transcripts[i]
    write_csv(csv_filepath, lines)
