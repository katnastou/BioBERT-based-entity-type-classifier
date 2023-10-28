#!/bin/bash

#run the script to do the spliting in n (here 300) files
mkdir -p split
python3 split_string.py -n 300 -s .tsv database_documents.tsv split/database_documents-
python3 split_string.py -n 300 -s .tsv database_matches.tsv split/database_matches-

#sort the split files
mkdir -p sorted-split
for f in split/database_documents-*.tsv; do time sort -n --parallel=40 $f > sorted-split/$(basename $f); done
for f in split/database_matches-*.tsv; do time sort -n --parallel=40 $f > sorted-split/$(basename $f); done

#do it separately for the organisms to make sure only predictions are only done for species

#keep only first matching line - have to do it before sorting to make sure it's the actual first one 
mkdir -p split-org-only
for i in {000..299}; do awk -F"\t" '$4=="-2"' split/database_matches-"$i".tsv > split-org-only/database_matches-"$i".tsv;done
mkdir split-org-only-first
for i in {000..299}; do awk -F"\t" '!seen[$1,$2,$3,$4]++' split-org-only/database_matches-"$i".tsv > split-org-only-first/database_matches-"$i".tsv; done


#add rank to matches --> keep species only and remove last column
mkdir -p split-org-only-first-ranked 
for i in {000..299}; do python3 add_rank_db_matches.py split-org-only-first/database_matches-"$i".tsv > split-org-only-first-ranked/database_matches-"$i".tsv; done

#keep species only
mkdir -p split-org-only-first-species
for i in {000..299}; do awk -F"\t" '$6=="species"{printf("%s\t%s\t%s\t%s\t%s\n",$1,$2,$3,$4,$5)}' split-org-only-first-ranked/database_matches-"$i".tsv > split-org-only-first-species/database_matches-"$i".tsv;done

#sort matches
mkdir -p sorted-split-org-only-first-species
for f in split-org-only-first-species/database_matches-*.tsv; do time sort -n --parallel=40 $f > sorted-split-org-only-first-species/$(basename $f); done

#get list of pmids for each of them and print the database documents
#I should keep documents after I have cleaned it down to species.
for i in {000..299}; do awk -F"\t" 'NR==FNR{a[$1];next}{if($1 in a){print $0}}' <(cut -f1 sorted-split-org-only-first-species/database_matches-"$i".tsv | sort -u ) sorted-split/database_documents-"$i".tsv > sorted-split-org-only-first-species/database_documents-"$i".tsv; done