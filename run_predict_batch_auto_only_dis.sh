#!/bin/bash


MAX_JOBS=300

mkdir -p output-biobert/predictions/dis-blocklists-auto-only


MODELS="
/scratch/project_2001426/katerina/output-biobert/multigpu/19143192/model.ckpt-48828
"

batch_size="32"

data_dir="/scratch/project_2001426/stringdata/blocklists-paper/auto-only/split-contexts"

type="consensus" 
labels_dir="data/biobert/other"

#all files from 000-299
NAME="dis-contexts-w100-[0-2][0-9][0-9].tsv"

PRED_DIR="/scratch/project_2001426/stringdata/blocklists-paper/auto-only/dis-predictions"


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
            while true; do
                jobs=$(ls output-biobert/predictions/dis-blocklists-auto-only | wc -l)
                if [ $jobs -lt $MAX_JOBS ]; then break; fi
                    echo "Too many jobs ($jobs), sleeping ..."
                    sleep 60
            done
            echo "Submitting job with params $model $dataset $max_seq_len $batch_size"
                job_id=$(
                sbatch slurm/slurm-run-predict.sh \
                    $config_dir \
                    $current_data_dir \
                    $max_seq_len \
                    $batch_size \
                    $type \
                    $model \
                    $labels_dir \
                    $PRED_DIR \
                    | perl -pe 's/Submitted batch job //'
                )
            echo "Submitted batch job $job_id"
            #change to base for base model
            touch output-biobert/predictions/dis-blocklists-auto-only/$job_id; 
            sleep 5
        done
    fi
done
