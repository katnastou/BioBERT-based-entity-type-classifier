#!/bin/bash

MAX_JOBS=200

mkdir -p output-biobert/finetuning

MODELS="
models/biobert_v1.1_pubmed
models/biobert_large
"

DATA_DIRS="
/scratch/project_2001426/data-may-2020/5-class-125K-w100-filtered-shuffled
"

BATCH_SIZES="32 64"

LEARNING_RATES="5e-5 3e-5 2e-5 1e-5 5e-6"

EPOCHS="2 3 4"

REPETITIONS=3

task="consensus"
labels_dir="data/biobert/other"

for repetition in `seq $REPETITIONS`; do
	for batch_size in $BATCH_SIZES; do
	    for learning_rate in $LEARNING_RATES; do
            for epochs in $EPOCHS; do
                for model in $MODELS; do
                    if [[ "$model" =~ "large" ]]; then
                        seq_len="96";
                        init_ckpt="models/biobert_large/bert_model.ckpt"
                    else
                        seq_len="256";
                        init_ckpt="models/biobert_v1.1_pubmed/model.ckpt-1000000"
                    fi
                    for data_dir in $DATA_DIRS; do
                        while true; do
                            jobs=$(ls output-biobert/finetuning/ | wc -l)
                            if [ $jobs -lt $MAX_JOBS ]; then break; fi
                            echo "Too many jobs ($jobs), sleeping ..."
                            sleep 60
                        done
                        echo "Submitting job with params $model $data_dir $seq_len $batch_size $learning_rate $epochs"
                        job_id=$(
                        sbatch slurm/slurm-run.sh \
                            $model \
                            $data_dir \
                            $seq_len \
                            $batch_size \
                            $learning_rate \
                            $epochs \
                            $task \
                            $init_ckpt \
                            $labels_dir \
                            | perl -pe 's/Submitted batch job //'
                        )
                        echo "Submitted batch job $job_id"
                        touch output-biobert/finetuning/$job_id
                        sleep 10
                    done
                done
            done
	    done
	done
done
