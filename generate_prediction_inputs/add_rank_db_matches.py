#!/usr/bin/env python3

import sys

from logging import warning


rank_by_taxid = {}


with open('scientific_names_and_ranks.tsv') as f:
    for ln, l in enumerate(f, start=1):
        l = l.rstrip('\n')
        fields = l.split('\t')
        id_, name, rank = fields
        rank_by_taxid[id_] = rank


for fn in sys.argv[1:]:
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            docid, start, end, type_, taxid = fields
            try:
                rank = rank_by_taxid[taxid]
            except KeyError:
                warning('missing taxid {}'.format(taxid))
                rank = 'unknown'
            fields.append(rank)
            print('\t'.join(fields))

