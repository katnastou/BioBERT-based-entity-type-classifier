#!/usr/bin/env python3

import sys
import os
from logging import warning

rank_by_taxid = {}

current_directory = os.path.dirname(__file__)  # Get the directory of the current script
path_to_file = os.path.join(current_directory, 'scientific_names_and_ranks.tsv')

if not os.path.isfile(path_to_file):  # If the file doesn't exist in the current directory, try one level above
    path_to_file = os.path.join(os.path.dirname(current_directory), 'scientific_names_and_ranks.tsv')

with open(path_to_file) as f:
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
