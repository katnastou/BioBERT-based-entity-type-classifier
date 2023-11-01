#!/bin/bash
# this script is the setup to run in prediction mode

# you need to set up tagger first in case you haven't already
# comment the next lines if you have already set up tagger
git clone https://github.com/larsjuhljensen/tagger tagger
cd tagger
#make 
make tagcorpus #to skip swig
cd ..

#download the dictionary files and the corpus
wget https://zenodo.org/api/records/10008720/files/dictionary-files-tagger-STRINGv12.zip?download=1
wget https://a3s.fi/s1000/PubMed-input.tar.gz
wget https://a3s.fi/s1000/PMC-OA-input.tar.gz
unzip dictionary-files-tagger-STRINGv12.zip 
tar -xzvf PubMed-input.tar.gz 
tar -xzvf PMC-OA-input.tar.gz 
rm *tar.gz *zip

mkdir -p results

gzip -cd `ls -1 ./pmc/*.en.merged.filtered.tsv.gz` `ls -1r ./pubmed/*.tsv.gz` | \
    cat dictionary-files-tagger-STRINGv12/excluded_documents.txt - | \
    tagger/tagcorpus \
    --threads=40 \ #change to number of threads available
    --autodetect \
    --types=dictionary-files-tagger-STRINGv12/curated_types.tsv \
    --entities=dictionary-files-tagger-STRINGv12/all_entities.tsv \
    --names=dictionary-files-tagger-STRINGv12/all_names_textmining.tsv \
    --groups=dictionary-files-tagger-STRINGv12/all_groups.tsv \
    --stopwords=dictionary-files-tagger-STRINGv12/curated_global.tsv \
    --local-stopwords=dictionary-files-tagger-STRINGv12/curated_local.tsv \
    --out-matches=results/all_matches.tsv \
    --out-segments=results/all_segments.tsv

#generate matches in the correct format with identifiers
./create_matches.pl dictionary-files-tagger-STRINGv12/all_entities.tsv results/all_matches.tsv results/database_matches.tsv

#generate documents in a tab delimited format
./create_documents.pl dictionary-files-tagger-STRINGv12/excluded_documents.txt 'gzip -cd `ls -1 pmc/*.en.merged.filtered.tsv.gz` `ls -1r pubmed/*.tsv.gz` |' results/database_documents.tsv

mkdir -p split
python3 split_string.py -n 300 -s .tsv results/database_documents.tsv split/database_documents- 
python3 split_string.py -n 300 -s .tsv results/database_matches.tsv split/database_matches-

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

mkdir -p delme
mkdir -p split-contexts

TYPES="
che
org
ggp
dis
"

for i in {000..299}; do
    for TYPE in $TYPES; do
        if [[ ${TYPE} != "org" ]]; then
            python3 get_contexts.py \
                -t ${TYPE} \
                -w 100 \
                sorted-split/database_{documents,matches}-$i.tsv \
                > split-contexts/${TYPE}-contexts-w100-$i.tsv \
                2>delme/${TYPE}_contexts-$i.txt
        else
            python3 get_contexts.py \
                -t ${TYPE} \
                -w 100 \
                sorted-split-org-only-first-species/database_{documents,matches}-$i.tsv \
                > split-contexts/${TYPE}-contexts-w100-$i.tsv \
                2>delme/${TYPE}_contexts-$i.txt
        fi
    done
done

mkdir -p models
wget https://zenodo.org/api/records/10008720/files/bert-base-finetuned-large-set.tar.gz?download=1
wget http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_v1.1_pubmed.tar.gz
tar -xzvf bert-base-finetuned-large-set.tar.gz -C models
tar -xzvf biobert_v1.1_pubmed.tar.gz -C models
rm *tar.gz 
INIT_CKPT="models/bert-base-finetuned-large-set/model.ckpt-48828"
BATCH_SIZE="32"

TASK="consensus" 
LABELS_DIR="data/biobert/other"
#base model
MAX_SEQ_LEN="256"
BERT_DIR="models/biobert_v1.1_pubmed"
cased="true"

if [ "$cased" = "true" ] ; then
    DO_LOWER_CASE=0
    CASING_DIR_PREFIX="cased"
    case_flag="--do_lower_case=False"
else
    DO_LOWER_CASE=1
    CASING_DIR_PREFIX="uncased"
    case_flag="--do_lower_case=True"
fi

OUTPUT_DIR="output-biobert/run"
mkdir -p $OUTPUT_DIR

for TYPE in $TYPES; do
    mkdir -p output-biobert/predictions/${TYPE}-blocklists-v12
    DATASET_DIR="split-contexts"
    #all files from 000-299
    NAME="${TYPE}-contexts-w100-[0-2][0-9][0-9].tsv"
    PRED_DIR="${TYPE}-predictions"
    for dataset in $(ls $DATASET_DIR); do
        #if dataset filename is between 00-49
        if [[ "${dataset##*./}" =~ ${NAME} ]]; then
            path_to_dataset="$DATASET_DIR/$dataset"
            basename_dir=`basename $dataset .tsv`
            current_data_dir="currentrun/$basename_dir"
            mkdir -p $current_data_dir
            cp $path_to_dataset "$current_data_dir/dev.tsv"
            cp $path_to_dataset "$current_data_dir/test.tsv"
            python3 run_ner_consensus.py \
                --do_prepare=true \
                --do_train=true \
                --do_eval=true \
                --do_predict=true \
                --replace_span="[unused1]" \
                --task_name=$TASK \
                --init_checkpoint=$INIT_CKPT \
                --vocab_file=$BERT_DIR/vocab.txt \
                --bert_config_file=$BERT_DIR/bert_config.json \
                --data_dir=$DATASET_DIR \
                --output_dir=$OUTPUT_DIR \
                --eval_batch_size=$BATCH_SIZE \
                --predict_batch_size=$BATCH_SIZE \
                --max_seq_length=$MAX_SEQ_LEN \
                --use_fp16 \
                --cased=$cased \
                --labels_dir=$LABELS_DIR #\
#                --use_xla \
#                --horovod \

            paste <(paste ${DATASET_DIR}"/test.tsv" ${OUTPUT_DIR}"/test_output_labels.txt") ${OUTPUT_DIR}"/test_results.tsv" | awk -F'\t' '{printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t\'\{\''che'\'': %s'\,' '\''dis'\'': %s'\,' '\''ggp'\'': %s'\,' '\''org'\'': %s'\,' '\''out'\'': %s'\}'\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' > ${OUTPUT_DIR}"/output_with_probabilities_dict.tsv"; 
            #remove everything up to last - 
            mkdir -p $PRED_DIR
            cp ${OUTPUT_DIR}"/output_with_probabilities_dict.tsv" ${PRED_DIR}"/output_with_probabilities_"$(basename ${DATASET_DIR##*-})".tsv"
            echo -n 'result written in '"$PRED_DIR"$'\n'
            sleep 5
        fi
    done
done

#after all prediction runs are finished you can generate blocklists

#generate local lists for everything
for type in $TYPES; do
    PRED_DIR="${type}-predictions"
    mkdir -p local_blacklists-01-${type}
    for f in ${PRED_DIR}/output_with_probabilities_*; do echo $f; time python3 ../blocklist_generation/make_local_blacklist.py -m 1 -t 1.01 -s ${type} -T ${type} $f > local_blacklists-01-${type}/$(basename $f); done


    if [ "$type" != "dis" ]; then
        python3 ../blocklist_generation/global_blacklist_candidates.py -m 2 -t 0.5 local_blacklists-01-${type}/output_with_probabilities_* | sort -rn > ${type}_global_min_mentions_1_min_docs_2_threshold_0.5.tsv
        awk -F"\t" '{printf("%s\tt\n",$4)}' ${type}_global_min_mentions_1_min_docs_2_threshold_0.5.tsv > global_blacklist_${type}_2col.tsv
    else
        #different threshold for diseases
        python3 ../blocklist_generation/global_blacklist_candidates.py -m 2 -t 0.85 local_blacklists-01-${type}/output_with_probabilities_* | sort -rn > ${type}_global_min_mentions_1_min_docs_2_threshold_0.85.tsv
        awk -F"\t" '{printf("%s\tt\n",$4)}' ${type}_global_min_mentions_1_min_docs_2_threshold_0.85.tsv > global_blacklist_${type}_2col.tsv
    fi

    if [[ "$type" == "che" || "$type" == "ggp" ]]; then
        python3 ../blocklist_generation/global_blacklist_candidates.py -m 2 -t 0.0 local_blacklists-01-${type}/*.tsv | sort -rn > global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv

        #local block/allowlist

        mkdir -p local-blacklists-ratio-1000-${type}
        for f in local_blacklists-01-${type}/*.tsv; do echo $f; python3 ../blocklist_generation/make_local_bw_lists.py --blacklist --ratio 1000 global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv $f | sort -rn > local-blacklists-ratio-1000-${type}/$(basename $f); done
        #allowlist/whitelist only for ggp
        if [ $type == "ggp" ] then
            mkdir -p local-whitelists-ratio-1e15-global-${type}-prob-0.3
            for f in local_blacklists-01-${type}/*.tsv; do echo $f; python3 ../blocklist_generation/make_local_bw_lists.py --ratio 1000000000000000 --global-ggp-prob 0.3 global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv $f | sort -rn > local-whitelists-ratio-1e15-global-${type}-prob-0.3/$(basename $f); done

            cat local-whitelists-ratio-1e15-global-${type}-prob-0.3/* > local-whitelist-${type}-ratio-1e15-global-${type}-prob-0.3.tsv

            awk -F"\t" '{printf("%s\t%s\tf\n",$2, $3)}' local-whitelist-${type}-ratio-1e15-global-${type}-prob-0.3.tsv > local_whitelist_${type}_3col.tsv
        fi

        cat local-blacklists-ratio-1000-${type}/* > local-blacklists-ratio-1000-${type}.tsv
        awk -F"\t" '{printf("%s\t%s\tt\n",$2, $3)}' local-blacklists-ratio-1000-${type}.tsv > local_blacklist_${type}_3col.tsv
    fi
done

#Concatenate all auto local lists
cat local_whitelist_ggp_3col.tsv local_blacklist_ggp_3col.tsv local_blacklist_che_3col.tsv | sort > auto_local.tsv

#Concatenate all automated global lists into one list
cat global_blacklist_che_2col.tsv global_blacklist_ggp_2col.tsv global_blacklist_dis_2col.tsv global_blacklist_org_2col.tsv | sort -u > auto_global.tsv
