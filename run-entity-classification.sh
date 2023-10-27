#!/bin/bash

BERT_DIR="models/biobert_v1.1_pubmed"
DATASET_DIR="5-class-125K-w100-filtered-shuffled"
MAX_SEQ_LENGTH="256"
BATCH_SIZE="32"
LEARNING_RATE="2e-5"
EPOCHS="4"
TASK="consensus"
INIT_CKPT="models/biobert_v1.1_pubmed/model.ckpt-1000000"
LABELS_DIR="data/biobert/other"


echo "data dir: $DATASET_DIR" 
echo "model: $BERT_DIR"


cased="true"

python3 run_ner_consensus.py \
    --do_prepare=true \
    --do_train=true \
    --do_eval=true \
    --do_predict=false \
    --replace_span="[unused1]" \
    --task_name=$TASK \
    --init_checkpoint=$INIT_CKPT \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --eval_batch_size=$BATCH_SIZE \
    --predict_batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ_LENGTH \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$EPOCHS \
    --cased=$cased \
    --labels_dir=$LABELS_DIR \
    --use_xla \
    --use_fp16 \
    --horovod


echo "Ready"