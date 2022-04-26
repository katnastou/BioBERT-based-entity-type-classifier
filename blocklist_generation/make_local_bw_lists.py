#!/usr/bin/env python3

import sys
import math

from logging import warning


# Manually validated cases
DO_NOT_BLACKLIST = set([
    ('insulin', 'ggp'),
    ('INSULIN', 'ggp'),
])


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-b', '--blacklist', default=False, action='store_true')
    ap.add_argument('-g', '--global-ggp-prob', type=float, default=None)
    ap.add_argument('-m', '--min-doc-count', default=10, type=int)
    ap.add_argument('-t', '--threshold', default=0.5, type=float)
    ap.add_argument('-f', '--function', default='ratio',
                    choices=['ratio', 'diff', 'simple'])
    ap.add_argument('-r', '--ratio', type=float, default=None)
    ap.add_argument('global_probs')
    ap.add_argument('local_probs', nargs='+')
    return ap


def load_global_probabilities(fn, options):
    probs, counts = {}, {}
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            try:
                exp_errors, count, avg_prob, name, type_ = fields
                exp_errors, avg_prob = float(exp_errors), float(avg_prob)
                count = int(count)
            except Exception as e:
                raise ValueError('line {} in {}: {}'.format(ln, fn, l))
            assert (name, type_) not in probs
            assert (name, type_) not in counts
            probs[(name, type_)] = avg_prob
            counts[(name, type_)] = count
    return probs, counts


# TODO this should be merged with global blacklist generation so that
# we first check if something is globally blacklisted, and only depending
# on that decision we decide whether it should be considered for local
# black/whitelisting.

def globally_blacklist(name, type_, count, non_ggp_prob, options):
    if (name, type_) not in globally_blacklist.cache:
        if count < options.min_doc_count:
            decision = False
        elif non_ggp_prob < options.threshold:
            decision = False
        elif (name, type_) in DO_NOT_BLACKLIST:
            print('NOTE: not blacklisting {}/{}'.format(name, type_),
                  file=sys.stderr)
            decision = False
        else:
            decision = True
        globally_blacklist.cache[(name, type)] = decision
    return globally_blacklist.cache[(name, type)]
globally_blacklist.cache = {}


def f_ratio(global_ggp_prob, local_ggp_prob, invert=False):
    if invert:
        global_ggp_prob = 1 - global_ggp_prob
        local_ggp_prob = 1 - local_ggp_prob        
    global_non_ggp_prob = 1 - global_ggp_prob
    local_non_ggp_prob = 1 - local_ggp_prob
    if local_non_ggp_prob * global_non_ggp_prob == 0:
        return math.inf
    else:
        return ((local_ggp_prob * global_ggp_prob) /
                (local_non_ggp_prob * global_non_ggp_prob))


def f_diff(global_ggp_prob, local_ggp_prob, invert=False):
    if invert:
        global_ggp_prob = 1 - global_ggp_prob
        local_ggp_prob = 1 - local_ggp_prob        
    global_non_ggp_prob = 1 - global_ggp_prob
    return local_ggp_prob - global_non_ggp_prob


def f_simple_ratio(global_ggp_prob, local_ggp_prob, invert=False):
    if invert:
        global_ggp_prob = 1 - global_ggp_prob
        local_ggp_prob = 1 - local_ggp_prob        
    global_non_ggp_prob = 1 - global_ggp_prob
    if global_non_ggp_prob == 0:
        return math.inf
    else:
        return local_ggp_prob / global_non_ggp_prob


def ranking_function(count, global_ggp_prob, local_ggp_prob, options,
                     invert=False):
    if options.function == 'ratio':
        ratio = f_ratio(global_ggp_prob, local_ggp_prob, invert)
        return '{}'.format(ratio)
    elif options.function == 'diff':
        diff = f_diff(global_ggp_prob, local_ggp_prob, invert)
        return '{:.5f}'.format(diff)
    elif options.function == 'simple':
        simple_ratio = f_diff(global_ggp_prob, local_ggp_prob, invert)
        return '{:.5f}'.format(simple_ratio)
    else:
        raise ValueError(options.function)
    

def make_bw_lists(fn, global_probs, global_counts, options):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            try:
                log_prob, count, doc_id, name, type_ = fields[:5]
                log_prob, count = float(log_prob), int(count)
            except Exception as e:
                raise ValueError('line {} in {}: {}'.format(ln, fn, l))
            local_ggp_prob = 10**log_prob
            local_non_ggp_prob = 1 - local_ggp_prob
            if (name, type_) not in global_probs:
                warning('no global probability for {}/{}'.format(name, type_))
                continue
            assert (name, type_) in global_counts
            global_non_ggp_prob = global_probs[(name, type_)]
            global_ggp_prob = 1 - global_non_ggp_prob
            doc_count = global_counts[(name, type_)]

            is_globally_blacklisted = globally_blacklist(
                name, type_, doc_count, global_non_ggp_prob, options)

            if is_globally_blacklisted and not options.blacklist:
                if ((local_ggp_prob > global_non_ggp_prob) and
                    (options.ratio is None or
                     f_ratio(global_ggp_prob, local_ggp_prob) >
                     options.ratio) and
                    (options.global_ggp_prob is None or
                     global_ggp_prob > options.global_ggp_prob)):
                    value = ranking_function(
                        count, global_ggp_prob, local_ggp_prob, options)
                    print('{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}'.format(
                        value, name, doc_id, count, global_ggp_prob,
                        local_ggp_prob))

            if options.blacklist and not is_globally_blacklisted:
                if ((local_non_ggp_prob > global_ggp_prob) and
                    (options.ratio is None or
                     f_ratio(global_ggp_prob, local_ggp_prob, True) >
                     options.ratio)):
                    value = ranking_function(
                        count, global_ggp_prob, local_ggp_prob, options, True)
                    print('{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}'.format(
                        value, name, doc_id, count, global_ggp_prob,
                        local_ggp_prob))

def main(argv):
    args = argparser().parse_args(argv[1:])
    global_probs, global_counts = load_global_probabilities(
        args.global_probs, args)
    for fn in args.local_probs:
        make_bw_lists(fn, global_probs, global_counts, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

