#!/usr/bin/env python3

import sys

from collections import Counter
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--min-count', default=2, type=int)
    ap.add_argument('gold')
    ap.add_argument('errors')
    return ap


def load_tsv(fn, field_num):
    data = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) != field_num:
                raise ValueError('Expected {} TAB-separated fields, got {}'
                                 ' on line {} of file {}: {}'.format(
                                     field_num, len(fields), ln, fn, l))
            data.append(fields)
    print('Read {} lines from {}'.format(len(data), fn), file=sys.stderr)
    return data

def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith('NN'):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
        else:
            yield word

def target_counts(data):
    counts = Counter()
    for fields in data:
        phrase = fields[4].lower()
        phrase = phrase.replace("this ","")
        phrase = phrase.replace("that ","")
        phrase = phrase.replace("those ","")
        phrase = phrase.replace("these ","")
        phrase = phrase.replace("the ","")
        phrase = phrase.replace("a ","")
        phrase = phrase.replace("other ","")
        phrase = ' '.join(lemmatize_all(phrase))
        type_, text = fields[2], phrase
        counts[(type_, text)] += 1
    return counts


def main(argv):
    args = argparser().parse_args(argv[1:])
    gold = load_tsv(args.gold, 6)
    errors = load_tsv(args.errors, 7)

    gold_count = target_counts(gold)
    error_count = target_counts(errors)

    error_freq = { k: error_count[k] / gold_count[k] for k in error_count }

    for (type_, text), ef in sorted(error_freq.items(), key=lambda i: -i[1]):
        gc, ec = gold_count[(type_, text)], error_count[(type_, text)]
        if ec >= args.min_count:
            print('{:.5f}\t{}\t{}\t{}\t{}'.format(ef, gc, ec, type_, text))
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

