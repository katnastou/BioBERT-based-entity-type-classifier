#!/usr/bin/env python3

import sys
import re

import itertools

from collections import defaultdict
from collections.abc import Iterator


EXTRA_SPACE_RE = re.compile(r'^\s+.*|.*\s+$')

TAB_UNESCAPE_RE = re.compile(r'(^|[^\\]|\\\\)\\t')

DOC_START = 'DOCSTART'

DOC_END = 'DOCEND'


# From https://bitbucket.org/larsjuhljensen/tagger/
TYPE_MAP = {
    -1: 'Chemical',
    -2: 'Organism',    # NCBI species taxonomy id (tagging species)
    -3: 'Organism',    # NCBI species taxonomy id (tagging proteins)
    -11: 'Wikipedia',
    -20: 'LifeStyle_Factor', #LFS
    -21: 'Biological_process',    # GO biological process
    -22: 'Cellular_component',    # GO cellular component
    -23: 'Molecular_function',    # GO molecular function
    -24: 'GO_other',    # GO other (unused)
    -25: 'Tissue',    # BTO tissues
    -26: 'Disease',    # DOID diseases
    -27: 'Environment',    # ENVO environments
    -28: 'Phenotype',    # APO phenotypes
    -29: 'Phenotype',    # FYPO phenotypes
    -30: 'Phenotype',    # MPheno phenotypes
    -31: 'Behaviour',    # NBO behaviors
    -36: 'Phenotype',    # mammalian phenotypes
    -50: 'Wikipedia_titles', #wikipedia article titles
}


CONSENSUS_TYPE_MAP = {
    'Gene': 'ggp',
    'Chemical': 'che',
    'Organism': 'org',
    'Disease': 'dis',
    'LifeStyle_Factor': 'lsf', #not in consensus
    'Wikipedia_titles': 'wiki',
}


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-m', '--max-docs', default=None, type=int,
                    help='maximum number of documents to process')
    ap.add_argument('-t', '--types', metavar='TYPE[,...]', default=None,
                    help='only include given types')
    ap.add_argument('-w', '--words', default=5, type=int,
                    help='number of context words to include')
    ap.add_argument('docs', help='documents in TAGGER TSV format')
    ap.add_argument('tags', help='tags in TAGGER TSV format')
    return ap


class LookaheadIterator(Iterator):
    """Lookahead iterator from http://stackoverflow.com/a/1518097."""

    def __init__(self, it):
        self._it, self._nextit = itertools.tee(iter(it))
        self.index = -1
        self._advance()

    def _advance(self):
        self.lookahead = next(self._nextit, None)
        self.index = self.index + 1

    def next(self):
        self._advance()
        return next(self._it)

    def __next__(self):
        return self.next()    # TODO cleanup

    def __nonzero__(self):
        return self.lookahead is not None

    def __bool__(self):
        return self.lookahead is not None


def normalize_space(s):
    return ' '.join(s.split())


def is_word(token):
    return any(c for c in token if c.isalnum())    # loose definition


def get_words(text, maximum, reverse=False):
    split = re.split(r'(\s+)', text)
    if reverse:
        split = reversed(split)
    words, count = [], 0
    for w in split:
        if count >= maximum:
            break
        words.append(w)
        if is_word(w):
            count += 1
    if reverse:
        words = reversed(words)
    return ''.join(words)


def get_context(text, start, end, words):
    before = normalize_space(DOC_START + ' ' + text[:start])
    after = normalize_space(text[end:] + ' ' + DOC_END)
    before = get_words(before, words, reverse=True)
    after = get_words(after, words, reverse=False)
    return before, after


def parse_tag_idx(line):
    line = line.rstrip('\n')
    fields = line.split('\t')
    try:
        idx = int(fields[0])
    except ValueError:
        raise ValueError('failed to parse {}'.format(line))
    return idx


def parse_doc_line(line):
    line = line.rstrip('\n')
    fields = line.split('\t')
    if len(fields) != 6:
        raise ValueError('expected 6 tab-separated fields, got {}'.format(
            len(fields)))
    idx, doc_id, authors, journal, year, text = fields
    idx = int(idx)
    text = TAB_UNESCAPE_RE.sub(r'\1''\t', text)
    text = text.replace(r'\\', '\\')
    #text = text.replace('<i>', '')
    #text = text.replace('&gt;', '>')
    #text = text.replace('&lt;', '<')
    #text = text.replace('&apos;', "'")
    #text = text.replace('&quot;', '"')
    #text = text.replace('&amp;', '&')
    # text = text.replace(r'\t', '\t').replace(r'\\', '\\')
    text = text.replace('\t', ' ')    # output format doesn't allow TABs
    return idx, text


def type_name(type_):
    if type_ > 0:
        return 'Gene'
    else:
        return TYPE_MAP.get(type_, '<UNKNOWN>')


def output_candidates(candidates, doc_id, doc_text, options):
    for (start, end, type_), data in candidates.items():
        type_ = CONSENSUS_TYPE_MAP.get(type_, 'out')
        if options.types is not None and type_ not in options.types:
            continue
        text = doc_text[start:end]
        before, after = get_context(doc_text, start, end, options.words)
        ann_indices = [str(i) for i, s in data]
        serials = [s for i, s in data]
        print('{}\t{}\t{}\t{}\t{}\t{}'.format(
            doc_id, ','.join(ann_indices), type_, before, text, after))


def update_document(doc_iter, idx):
    doc_idx, doc_text = parse_doc_line(doc_iter.lookahead)
    while doc_idx < idx:
        doc_idx, doc_text = parse_doc_line(next(doc_iter))
    if doc_idx > idx:
        raise ValueError('missing document {}'.format(idx))
    return doc_idx, doc_text


def document_annotations(df, tf, options):
    doc_iter = LookaheadIterator(df)
    tag_iter = LookaheadIterator(tf)
    if not doc_iter:
        raise ValueError('no documents')
    if not tag_iter:
        raise ValueError('no tags')

    curr_idx = parse_tag_idx(tag_iter.lookahead)
    curr_ann, ann_idx, curr_ok = defaultdict(list), 1, True
    doc_idx, doc_text = update_document(doc_iter, curr_idx)
    doc_count = 0
    for tag_ln, tag_line in enumerate(tag_iter, start=1):
        if options.max_docs is not None and doc_count >= options.max_docs:
            break
        tag_line = tag_line.rstrip('\n')
        fields = tag_line.split('\t')
        idx, start, end, type_, serial = fields
        idx, start, end, type_ = (
            int(i) for i in [idx, start, end, type_]
        )
        end = end + 1    # adjust inclusive to exclusive
        type_str = type_name(type_)
        if idx != curr_idx:
            yield curr_idx, doc_text, curr_ann, curr_ok
            curr_ann, ann_idx, curr_ok = defaultdict(list), 1, True
            curr_idx = idx
            doc_count += 1
        if doc_idx < idx:
            doc_idx, doc_text = update_document(doc_iter, idx)
        span_text = doc_text[start:end]
        if EXTRA_SPACE_RE.match(span_text):
            print('extra space in {} (skip doc) ({}-{}): "{}" (...{})'.format(
                curr_idx, start, end, span_text, doc_text[start-100:start]),
                  file=sys.stderr)
            curr_ok = False
        curr_ann[(start, end, type_str)].append((ann_idx, serial))
        ann_idx += 1
    if options.max_docs is None or doc_count < options.max_docs:
        yield curr_idx, doc_text, curr_ann, curr_ok


def get_contexts(df, tf, options):
    for idx, text, anns, is_ok in document_annotations(df, tf, options):
        if is_ok:
            output_candidates(anns, idx, text, options)


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.types is not None:
        args.types = set(args.types.split(','))
    with open(args.docs) as df:
        with open(args.tags) as tf:
            get_contexts(df, tf, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
