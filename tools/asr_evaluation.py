import argparse

from uetasr.utils.evaluation import evaluate


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR transcript against a reference transcript.'
    )
    parser.add_argument(
        'ref',
        type=str,
        help='Reference transcript filepath.'
    )
    parser.add_argument(
        'hyp',
        type=str,
        help='ASR hypothesis filepath.'
    )
    parser.add_argument(
        '--case-insensitive',
        action='store_true',
        help='Down-case the text before running the evaluation.'
    )
    parser.add_argument(
        '--skip-empty-refs',
        action='store_true',
        help='Skip empty references.'
    )
    parser.add_argument(
        '--confusions',
        action='store_true',
        help='Print tables of which words were confused.'
    )
    parser.add_argument(
        '--min-word-count',
        type=int,
        default=1,
        help='Minimum word count to show a word in confusions (default 1).'
    )
    parser.add_argument(
        '--wer-vs-length',
        action='store_true',
        help='Print a table of WER vs. reference length.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    evaluate(args)
