#!/bin/bash

MAX_JOBS=300

MODELS="
/scratch/project_2001426/katerina/output-biobert/multigpu/19143192/model.ckpt-48828
"

TYPES="
che
org
ggp
dis
"

batch_size="32"

data_dir="/scratch/project_2001426/stringdata/STRING-blocklists-v12/split-contexts"

task="consensus" 
labels_dir="data/biobert/other"

#all files from 000-299

for TYPE in $TYPES; do

    NAME="${TYPE}-contexts-w100-[0-2][0-9][0-9].tsv"
    PRED_DIR="/scratch/project_2001426/stringdata/STRING-blocklists-v12/${TYPE}-predictions"
    JOB_DIR="output-biobert/predictions/${TYPE}-blocklists-v12"
    mkdir -p "$JOB_DIR"
    
    for dataset in $(ls $data_dir); do
        #if dataset filename is between 00-49
        if [[ "${dataset##*./}" =~ ${NAME} ]]; then
            path_to_dataset="$data_dir/$dataset"
            basename_dir=`basename $dataset .tsv`
            current_data_dir="/scratch/project_2001426/stringdata/currentrun/$basename_dir"
            mkdir -p $current_data_dir
            cp $path_to_dataset "$current_data_dir/dev.tsv"
            cp $path_to_dataset "$current_data_dir/test.tsv"
            for model in $MODELS; do
                #base model
                max_seq_len="256"
                config_dir="models/biobert_v1.1_pubmed"
                while true; do
                    jobs=$(ls ${JOB_DIR} | wc -l)
                    if [ $jobs -lt $MAX_JOBS ]; then break; fi
                        echo "Too many jobs ($jobs), sleeping ..."
                        sleep 60
                done
                echo "Submitting job with params $model $dataset $max_seq_len $batch_size $TYPE"
                    job_id=$(
                    sbatch slurm/slurm-run-predict.sh \
                        $config_dir \
                        $current_data_dir \
                        $max_seq_len \
                        $batch_size \
                        $task \
                        $model \
                        $labels_dir \
                        $PRED_DIR \
                        $JOB_DIR \
                        | perl -pe 's/Submitted batch job //'
                    )
                echo "Submitted batch job $job_id"
                touch "${JOB_DIR}/${job_id}"; 
                sleep 5
            done
        fi
    done
done