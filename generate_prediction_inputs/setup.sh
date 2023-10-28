#!/bin/bash

#get tagger results with global dictionary
# you need to set up tagger first in case you haven't already
# comment the next lines if you have already set up tagger
git clone https://github.com/larsjuhljensen/tagger tagger
cd tagger
make
cd ..

#download the dictionary files and the corpus
wget https://zenodo.org/api/records/10008720/files/dictionary-files-tagger-STRINGv12.zip?download=1
wget https://a3s.fi/s1000/PubMed-input.tar.gz
wget https://a3s.fi/s1000/PMC-OA-input.tar.gz
unzip dictionary-files-tagger-STRINGv12.zip 
tar -xzvf PubMed-input.tar.gz 
tar -xzvf PMC-OA-input.tar.gz 
rm *tar.gz *zip

#run tagger with curated blocklists only
gzip -cd `ls -1 pmc/*.en.merged.filtered.tsv.gz` `ls -1r pubmed/*.tsv.gz` | cat dictionary-files-tagger-STRINGv12/excluded_documents.txt - | tagger/tagcorpus --threads=40 --autodetect --types=dictionary-files-tagger-STRINGv12/curated_types.tsv --entities=dictionary-files-tagger-STRINGv12/all_entities.tsv --names=dictionary-files-tagger-STRINGv12/all_names_textmining.tsv --groups=dictionary-files-tagger-STRINGv12/all_groups.tsv --stopwords=curated_global.tsv --local-stopwords==curated_local.tsv --out-matches=all_prediction_matches.tsv --out-segments=all_prediction_segments.tsv 

#generate matches in the correct format with identifiers
./create_matches.pl dictionary-files-tagger-STRINGv12/all_entities.tsv all_prediction_matches.tsv database_prediction_matches.tsv

#generate documents in a tab delimited format
./create_documents.pl tagger-all-dictionary/excluded_documents.txt database_documents.tsv

#run the sort and split of matches and documents
./sort-split.sh &
wait

#generate the contexts for the prediction runs 
./run-context-split.sh &
wait

#next step is to invoke the slurm scripts run_predict_batch_auto* to generate the model predictions
bash ../run_predict_batch_auto_che.sh &
wait
bash ../run_predict_batch_auto_dis.sh & 
wait
bash ../run_predict_batch_auto_ggp.sh & 
wait
bash ../run_predict_batch_auto_org.sh & 
wait

#after all sbatch prediction runs are finished go to blocklist_generation and run ./generate-blocklists.sh to generate blocklists for each class

