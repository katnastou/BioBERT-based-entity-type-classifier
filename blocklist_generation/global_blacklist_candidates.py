#!/usr/bin/env python3
#author: Sampo Pyysalo, @github:spyysalo
# Suggest candidates for global blacklist based on local blacklists

import sys

from collections import defaultdict


# Manually validated cases
DO_NOT_BLACKLIST = set([
    ('insulin', 'ggp'),
    ('INSULIN', 'ggp'),
])


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-m', '--min-count', default=10, type=int)
    ap.add_argument('-t', '--threshold', default=0.5, type=float)
    ap.add_argument('file', nargs='+')
    return ap


def process_local_blacklist(fn, probs_by_name_and_type, options):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            log_prob, count, doc_id, name, type_ = fields[:5]
            log_prob, count = float(log_prob), int(count)
            prob = 1 - 10**log_prob
            probs_by_name_and_type[(name, type_)].append(prob)
    return probs_by_name_and_type


def main(argv):
    args = argparser().parse_args(argv[1:])
    probs_by_name_and_type = defaultdict(list)
    for fn in args.file:
        process_local_blacklist(fn, probs_by_name_and_type, args)
    for (name, type_), probs in probs_by_name_and_type.items():
        count = len(probs)
        if count < args.min_count:
            continue
        avg_prob = sum(probs)/count
        if avg_prob < args.threshold:
            continue
        if (name, type_) in DO_NOT_BLACKLIST:
            print('NOTE: not suggesting {}/{}'.format(name, type_),
                  file=sys.stderr)
            continue
        expected_errors = avg_prob * count
        print('{}\t{}\t{}\t{}\t{}'.format(
            expected_errors, count, avg_prob, name, type_))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
