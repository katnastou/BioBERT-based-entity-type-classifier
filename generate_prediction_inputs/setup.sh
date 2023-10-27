#!/bin/bash

#get tagger results with global dictionary
# you need to set up tagger first in case you haven't already
# uncomment the next lines to do so
# git clone https://github.com/larsjuhljensen/tagger tagger
# cd tagger
# make
# cd ..

#the dictionary files and the corpus can be downloaded from https://doi.org/10.5281/zenodo.10008720
wget <add zenodo link with dictionary files>
wget https://a3s.fi/s1000/PubMed-input.tar.gz
wget https://a3s.fi/s1000/PMC-OA-input.tar.gz
tar -xzvf tagger-all-dictionary.tar.gz 
tar -xzvf PubMed-input.tar.gz 
tar -xzvf PMC-OA-input.tar.gz 
rm *tar.gz

gzip -cd `ls -1 pmc/*.en.merged.filtered.tsv.gz` `ls -1r pubmed/*.tsv.gz` | cat tagger-all-dictionary/excluded_documents.txt - | tagger/tagcorpus --threads=40 --autodetect --types=tagger-all-dictionary/curated_types.tsv --entities=tagger-all-dictionary/all_entities.tsv --names=tagger-all-dictionary/all_names_textmining.tsv --groups=tagger-all-dictionary/all_groups.tsv --stopwords=curated_global.tsv --local-stopwords==curated_local.tsv --out-matches=all_prediction_matches.tsv --out-segments=all_prediction_segments.tsv 


./create_matches.pl all_prediction_matches.tsv database_prediction_matches.tsv

#check if that works
python3 ./add_rank_db_matches.py database_prediction_matches.tsv > database_prediction_matches_with_txids_and_ranks.tsv

