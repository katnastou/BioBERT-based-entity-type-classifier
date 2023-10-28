#!/usr/bin/env python3

import sys
import os


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-n', '--number', type=int, default=100)    
    ap.add_argument('-s', '--suffix', default='')
    ap.add_argument('data', help='Data in TAGGER TSV format')
    ap.add_argument('prefix', help='Output file prefix')
    return ap


def open_output(index, options):
    mn = len(str(options.number-1))
    fn = '{}{}{}'.format(options.prefix, str(index).zfill(mn), options.suffix)
    return open(fn, 'wt')


def split(fn, options):
    out = [open_output(i, options) for i in range(options.number)]
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            doc_id = int(fields[0])
            index = doc_id % options.number
            print(l, file=out[index])
    for f in out:
        f.close()
            

def main(argv):
    args = argparser().parse_args(argv[1:])
    split(args.data, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
