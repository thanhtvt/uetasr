"""
Adapted from: https://github.com/belambert/asr-evaluation/
"""
import os
import warnings
from collections import defaultdict
from edit_distance import SequenceMatcher
from functools import reduce
from typing import List


# For keeping track of the total number of tokens, errors, and matches
ref_token_count = 0
error_count = 0
match_count = 0
counter = 0
sent_error_count = 0

# Save number of errors
total_ins = 0
total_dels = 0
total_subs = 0

# For keeping track of word error rates by sentence length
# this is so we can see if performance is better/worse for longer
# and/or shorter sentences
error_analysis = []
lengths = []
error_rates = []
wer_bins = defaultdict(list)
wer_vs_length = defaultdict(list)
# Tables for keeping track of which words get confused with one another
insertion_table = defaultdict(int)
deletion_table = defaultdict(int)
substitution_table = defaultdict(int)
# These are the editdistance opcodes that are condsidered 'errors'
error_codes = ['replace', 'delete', 'insert']


def read_file(filepath):
    """Read a file into a list of lines, and return the list."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines


def evaluate(args):
    """Main method - this reads the hyp and ref files, and creates
    editdistance.SequenceMatcher objects to compute the edit distance.
    All the statistics necessary statistics are collected, and results are
    printed as specified by the command line options.
    """
    global counter
    global min_count
    # Put the command line options into global variables.
    min_count = args.min_word_count

    counter = 0

    # Read the files
    refs = read_file(args.ref)
    hyps = read_file(args.hyp)

    # Check that the files are the same length
    if len(refs) != len(hyps):
        warnings.warn(
            'The reference and hypothesis files have different numbers of lines.'
        )

    for ref_line, hyp_line in zip(refs, hyps):
        processed_p = process_line_pair(ref_line,
                                        hyp_line,
                                        args.case_insensitive,
                                        args.skip_empty_refs,
                                        args.confusions)
        if processed_p:
            counter += 1

    if args.confusions:
        confusion_file = os.path.dirname(args.ref) + '/confusion.txt'
        print_confusions(confusion_file)
    if args.wer_vs_length:
        wer_vs_length_file = os.path.dirname(args.ref) + '/wer_vs_length.txt'
        print_wer_vs_length(wer_vs_length_file)

    # Compute WER and WRR
    if ref_token_count > 0:
        wrr = match_count / ref_token_count
        wer = error_count / ref_token_count
    else:
        wrr = 0.0
        wer = 0.0
    # Compute SER
    ser = sent_error_count / counter if counter > 0 else 0.0

    error_log_file = os.path.dirname(args.ref) + '/error_log.txt'
    print_error_log(error_log_file, wer, wrr, ser)


def process_line_pair(
    ref_line: str,
    hyp_line: str,
    case_insensitive: bool = False,
    remove_empty_refs: bool = False,
    confusions: bool = False,
):
    """Given a pair of strings corresponding to a reference and hypothesis,
    compute the edit distance, print if desired, and keep track of results
    in global variables.

    Return true if the pair was counted, false if the pair was not counted due
    to an empty reference string."""

    global error_count
    global match_count
    global ref_token_count
    global sent_error_count

    # Split into tokens by whitespace
    ref = ref_line.split()
    hyp = hyp_line.split()

    if case_insensitive:
        ref = list(map(str.lower, ref))
        hyp = list(map(str.lower, hyp))
    if remove_empty_refs and len(ref) == 0:
        return False

    # Create an object to get the edit distance, and then retrieve the
    # relevant counts that we need.
    sm = SequenceMatcher(a=ref, b=hyp)
    errors = get_error_count(sm)
    matches = get_match_count(sm)
    ref_length = len(ref)

    # Increment the total counts we're tracking
    error_count += errors
    match_count += matches
    ref_token_count += ref_length

    if errors != 0:
        sent_error_count += 1

    # If we're keeping track of which words get mixed up with which others
    # call track_confusions
    if confusions:
        track_confusions(sm, ref, hyp)

    print_instances(ref, hyp, sm)

    # Keep track of the individual error rates, and reference lengths, so we
    # can compute average WERs by sentence length
    lengths.append(ref_length)
    error_rate = errors * 1.0 / len(ref) if len(ref) > 0 else float("inf")
    error_rates.append(error_rate)
    wer_bins[len(ref)].append(error_rate)
    return True


def get_match_count(sm: SequenceMatcher):
    "Return the number of matches, given a sequence matcher object."
    matches = None
    matches1 = sm.matches()
    matching_blocks = sm.get_matching_blocks()
    matches2 = reduce(lambda x, y: x + y, [x[2] for x in matching_blocks], 0)
    assert matches1 == matches2
    matches = matches1
    return matches


def get_error_count(sm: SequenceMatcher):
    """Return the number of errors (insertion, deletion, and substitutiions
    , given a sequence matcher object."""
    opcodes = sm.get_opcodes()
    errors = [x for x in opcodes if x[0] in error_codes]
    error_lengths = [max(x[2] - x[1], x[4] - x[3]) for x in errors]
    return reduce(lambda x, y: x + y, error_lengths, 0)


def track_confusions(sm: SequenceMatcher, seq1: List[str], seq2: List[str]):
    """Keep track of the errors in a global variable, given a sequence matcher."""
    opcodes = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'insert':
            for i in range(j1, j2):
                word = seq2[i]
                insertion_table[word] += 1
        elif tag == 'delete':
            for i in range(i1, i2):
                word = seq1[i]
                deletion_table[word] += 1
        elif tag == 'replace':
            for w1 in seq1[i1:i2]:
                for w2 in seq2[j1:j2]:
                    key = (w1, w2)
                    substitution_table[key] += 1


def print_confusions(filepath: str):
    """Print the confused words that we found... grouped by insertions, deletions
    and substitutions."""
    confusions = []
    if len(insertion_table) > 0:
        confusions.append('INSERTIONS:')
        for item in sorted(list(insertion_table.items()), key=lambda x: x[1], reverse=True):
            if item[1] >= min_count:
                confusions.append('{0:20s} {1:10d}'.format(*item))
    if len(deletion_table) > 0:
        confusions.append('DELETIONS:')
        for item in sorted(list(deletion_table.items()), key=lambda x: x[1], reverse=True):
            if item[1] >= min_count:
                confusions.append('{0:20s} {1:10d}'.format(*item))
    if len(substitution_table) > 0:
        confusions.append('SUBSTITUTIONS:')
        for [w1, w2], count in sorted(list(substitution_table.items()), key=lambda x: x[1], reverse=True):
            if count >= min_count:
                confusions.append('{0:20s} -> {1:20s}   {2:10d}'.format(w1, w2, count))

    with open(filepath, 'w') as f:
        f.write('\n'.join(confusions))


def print_instances(ref: List[str],
                    hyp: List[str],
                    sm: SequenceMatcher):
    """Print a single instance of a ref/hyp pair."""
    global total_dels
    global total_ins
    global total_subs

    n_dels, n_subs, n_ins = get_diff(sm, ref, hyp)
    total_dels += n_dels
    total_ins += n_ins
    total_subs += n_subs

    error_analysis.append("SENTENCE {0:d}".format(counter + 1))

    # Handle cases where the reference is empty without dying
    if len(ref) != 0:
        correct_rate = sm.matches() / len(ref)
        error_rate = sm.distance() / len(ref)
    elif sm.matches() == 0:
        correct_rate = 1.0
        error_rate = 0.0
    else:
        correct_rate = 0.0
        error_rate = sm.matches()

    error_analysis.append('Correct       = {0:6.1%}  ({1:3d}   /  {2:3d})'.format(correct_rate, sm.matches(), len(ref)))
    error_analysis.append('Errors        = {0:6.1%}  ({1:3d}   /  {2:3d}, {3:3d} dels  , {4:3d} subs  , {5:3d} ins )\n'.format(error_rate, sm.distance(), len(ref), n_dels, n_subs, n_ins))


def get_diff(sm: SequenceMatcher, seq1: List[str], seq2: List[str]):
    """Given a sequence matcher and the two sequences, print a Sphinx-style
    'diff' off the two."""
    ref_tokens = []
    err_tokens = []
    hyp_tokens = []

    ndels, nsubs, nins = 0, 0, 0

    opcodes = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for i in range(i1, i2):
                ref_tokens.append(seq1[i].lower())

                # insert '=' at the middle of the error
                spaces = (len(seq1[i]) - 1) // 2
                err_tokens.append(' ' * spaces + '=' + ' ' * (len(seq1[i]) - spaces - 1))

            for i in range(j1, j2):
                hyp_tokens.append(seq2[i].lower())

        elif tag == 'delete':
            ndels += (i2 - i1)
            for i in range(i1, i2):
                ref_tokens.append(seq1[i].upper())

                # insert 'D' at the middle of the error
                spaces = (len(seq1[i]) - 1) // 2
                err_tokens.append(' ' * spaces + 'D' + ' ' * (len(seq1[i]) - spaces - 1))

            for i in range(i1, i2):
                hyp_tokens.append('*' * len(seq1[i]))
        elif tag == 'insert':
            nins += (j2 - j1)
            for i in range(j1, j2):
                ref_tokens.append('*' * len(seq2[i]))

                # insert 'I' at the middle of the error
                spaces = (len(seq2[i]) - 1) // 2
                err_tokens.append(' ' * spaces + 'I' + ' ' * (len(seq2[i]) - spaces - 1))

            for i in range(j1, j2):
                hyp_tokens.append(seq2[i].upper())

        elif tag == 'replace':
            seq1_len = i2 - i1
            seq2_len = j2 - j1
            # Get a list of tokens for each
            s1 = list(map(str.upper, seq1[i1:i2]))
            s2 = list(map(str.upper, seq2[j1:j2]))
            # Pad the two lists with False values to get them to the same length
            if seq1_len > seq2_len:
                for i in range(0, seq1_len - seq2_len):
                    s2.append(False)
            if seq1_len < seq2_len:
                for i in range(0, seq2_len - seq1_len):
                    s1.append(False)
            assert len(s1) == len(s2)
            # Pair up words with their substitutions, or fillers
            for i in range(0, len(s1)):
                w1 = s1[i]
                w2 = s2[i]
                # If we have two words, make them the same length
                if w1 and w2:
                    if len(w1) > len(w2):
                        s2[i] = w2 + ' ' * (len(w1) - len(w2))
                    elif len(w1) < len(w2):
                        s1[i] = w1 + ' ' * (len(w2) - len(w1))
                # Otherwise, create an empty filler word of the right width
                if not w1:
                    s1[i] = '*' * len(w2)
                if not w2:
                    s2[i] = '*' * len(w1)

            ref_tokens += s1
            hyp_tokens += s2
            nsubs += len(s1)
            for s in s1:
                spaces = (len(s) - 1) // 2
                err_tokens.append(' ' * spaces + 'S' + ' ' * (len(s) - spaces - 1))

    ref_tokens.insert(0, 'REF:')
    err_tokens.insert(0, '    ')
    hyp_tokens.insert(0, 'HYP:')

    error_analysis.append("=" * 20)
    error_analysis.append(' '.join(ref_tokens))
    error_analysis.append(' '.join(err_tokens))
    error_analysis.append(' '.join(hyp_tokens))

    return ndels, nsubs, nins


def print_wer_vs_length(filepath: str):
    """Print the average word error rate for each length sentence."""
    wer_vs_length_content = []

    avg_wers = {length: mean(wers) for length, wers in wer_bins.items()}
    for length, avg_wer in sorted(avg_wers.items(), key=lambda x: (x[1], x[0])):
        wer_vs_length_content.append('{0:5d} {1:f}'.format(length, avg_wer))

    # write to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(wer_vs_length_content))


def mean(seq):
    """Return the average of the elements of a sequence."""
    return float(sum(seq)) / len(seq) if len(seq) > 0 else float('nan')


def print_error_log(filepath: str, wer: float, wrr: float, ser: float):
    with open(filepath, 'w') as f:
        f.write('Sentence count: {}\n'.format(counter))
        f.write('WER: {:10.3%} ({:10d} / {:10d}, {:10d} dels, {:10d} subs, {:10d} ins)\n'.format(wer, error_count, ref_token_count, total_dels, total_subs, total_ins))
        f.write('WRR: {:10.3%} ({:10d} / {:10d})\n'.format(wrr, match_count, ref_token_count))
        f.write('SER: {:10.3%} ({:10d} / {:10d})\n'.format(ser, sent_error_count, counter))
        f.write('--------------------\n\n')
        f.write('\n'.join(error_analysis))
