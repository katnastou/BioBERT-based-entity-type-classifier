#!/usr/bin/env python3

# Make local blacklist based on classifier predictions

import sys
import json

from math import log10
from collections import Counter, defaultdict


LOG10_ZERO = -1000    # for numeric underflow


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-m', '--min-count', default=2, type=int)
    ap.add_argument('-p', '--probability', default=False, action='store_true',
                    help='Output probability instead of log-probability')
    ap.add_argument('-t', '--threshold', default=0.5, type=float)
    ap.add_argument('-s', '--source-type', default='ggp')
    ap.add_argument('-T', '--target-type', default='ggp')
    ap.add_argument('-v', '--verbose', default=False, action='store_true')
    ap.add_argument('file', nargs='+')
    return ap


def exp_normalize(log_prob_sums, base=10):
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    m = max(log_prob_sums.values())
    s = { k: base**(v-m) for k, v in log_prob_sums.items() }
    t = sum(s.values())
    n = { k: v/t for k, v in s.items() }
    return n


def sum_log_probs(probs_by_type):
    log_prob_sums = {}
    for type_, probs in probs_by_type.items():
        log_probs = [log10(p) for p in probs]
        log_prob_sums[type_] = sum(log_probs)
    return log_prob_sums


def dict_argmax(d):
    return max(d.items(), key=lambda i: i[1])[0]


def write_local_blacklist(doc_id, probs_by_name_and_type, options):
    for name, probs_by_type in probs_by_name_and_type.items():
        log_prob_sums = sum_log_probs(probs_by_type)
        normalized = exp_normalize(log_prob_sums)
        target_prob = normalized[options.target_type]
        most_likely = dict_argmax(normalized)
        count = len(probs_by_type[options.target_type])
        if target_prob < options.threshold and count >= options.min_count:
            if options.verbose:
                details = '\t{}\t{:.2f}'.format(most_likely, normalized[most_likely])
                details += '\t{}'.format(probs_by_type[options.target_type])
            else:
                details = ''
            if target_prob > 0:
                log_prob = log10(target_prob)
            else:
                log_prob = LOG10_ZERO
            if options.probability:
                prob_str = '{:.5f}'.format(10**log_prob)
            else:
                prob_str = '{}'.format(log_prob)
            print('{}\t{}\t{}\t{}\t{}{}'.format(prob_str, count, doc_id, name, options.target_type, details))


def update_probs(log_prob_sums, probs, s):
    log_probs = { k: log10(v) for k, v in probs.items() }
    if s not in log_prob_sums:
        log_prob_sums[s] = log_probs
    else:
        for k, v in log_probs.items():
            log_prob_sums[s][k] += v


def store_probs(probs_by_name_and_type, probs_by_type, name):
    if name not in probs_by_name_and_type:
        probs_by_name_and_type[name] = defaultdict(list)
    for k, v in probs_by_type.items():
        probs_by_name_and_type[name][k].append(v)


def process_predictions(fn, options):
    curr_doc_id, probs_by_name_and_type = None, None
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) != 8:
                raise ValueError('line {} in {}: {}'.format(ln, fn, l))
            doc_id, ann_id, s_type, before, name, after, p_type, prob = fields
            try:
                probs_by_type = json.loads(prob.replace("'", '"'))
            except:
                raise ValueError('line {} in {}: {}'.format(ln, fn, prob))
            if doc_id != curr_doc_id:
                if curr_doc_id is not None:
                    write_local_blacklist(curr_doc_id, probs_by_name_and_type,
                                          options)
                curr_doc_id = doc_id
                probs_by_name_and_type = {}
            if s_type == options.source_type:
                store_probs(probs_by_name_and_type, probs_by_type, name)
    if curr_doc_id is not None:
        write_local_blacklist(curr_doc_id, probs_by_name_and_type, options)
        

def main(argv):
    args = argparser().parse_args(argv[1:])
    for fn in args.file:
        process_predictions(fn, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
