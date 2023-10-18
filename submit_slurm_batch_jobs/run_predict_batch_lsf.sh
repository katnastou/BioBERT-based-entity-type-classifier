#!/bin/bash

MAX_JOBS=110
#output-biobert is a symlink to the project output folder
#mkdir -p output-biobert/predictions/large
mkdir -p output-biobert/predictions/base_new_week35

#large trained model: 2595163/model.ckpt-74483
#base trained model: 2603931/model.ckpt-74483
#better base model 2699830/model.ckpt-74483
#/scratch/project_2001426/output-biobert/multigpu/2595163/model.ckpt-74483

MODELS="
/scratch/project_2001426/katerina/output-biobert/multigpu/2699830/model.ckpt-48828
"

batch_size="32"

data_dir="/scratch/project_2001426/stringdata/week_35/split-contexts/lsf-to-other-label"

type="consensus" 
labels_dir="data/biobert/other"

#all files from 00-49
NAME="lsf-contexts-w100-[0-9][0-9].tsv"

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
            if [[ "$model" =~ "2595163" ]]; then
                #large model
                config_dir="models/biobert_large"
                max_seq_len="96";
                while true; do
                #change to base for base model
                    jobs=$(ls output-biobert/predictions/large | wc -l)
                    if [ $jobs -lt $MAX_JOBS ]; then break; fi
                        echo "Too many jobs ($jobs), sleeping ..."
                        sleep 60
                done
            else
                #base model
                max_seq_len="256"
                config_dir="models/biobert_v1.1_pubmed"
                while true; do
                #change to base for base model
                    jobs=$(ls output-biobert/predictions/base_new_week35 | wc -l)
                    if [ $jobs -lt $MAX_JOBS ]; then break; fi
                        echo "Too many jobs ($jobs), sleeping ..."
                        sleep 60
                done
            fi
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
                    | perl -pe 's/Submitted batch job //'
                )
            echo "Submitted batch job $job_id"
            #change to base for base model
            if [[ "$model" =~ "2595163" ]]; then touch output-biobert/predictions/large/$job_id; fi
            sleep 2
            if [[ "$model" =~ "2699830" ]]; then touch output-biobert/predictions/base_new_week35/$job_id; fi
            sleep 2
        done
    fi
done
