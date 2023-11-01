#!/bin/bash

BERT_DIR="models/biobert_v1.1_pubmed"
DATASET_DIR="data/biobert/example_data"
MAX_SEQ_LENGTH="256"
BATCH_SIZE="32"
LEARNING_RATE="2e-5"
EPOCHS="4"
TASK="consensus"
INIT_CKPT="models/biobert_v1.1_pubmed/model.ckpt-1000000"
LABELS_DIR="data/biobert/other"
OUTPUT_DIR="output-biobert"
mkdir -p $OUTPUT_DIR

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


# where `replace_span` is the span to replace the named entity in focus, in case of masking (no replacement is done if left undefined), 
# `task_name` is the name of the NER task (consensus in this case), 
# `init_checkpoint` is the checkpoint of the model (should change to the trained model during prediction), 
# `vocab_file` is the vocabulary file of BERT (same directory as the original model), 
# `bert_config_file` is the configuration json of the model, 
# `data_dir` is the directory with input data, 
# `output_dir` is the directory with output data, 
# `eval_batch_size` and `predict_batch_size` is the batch size, 
# `max_seq_length` is the maximum sequence length in tokens, 
# `learning_rate`, `num_training_epochs` are the initial learning rate and the maximum number of training epochs, 
# `cased` is the flag to define whether to run with lower case or not, based on the model, 
# `labels_dir` is the directory with the labels file containing the labels for the supervised training task 
# the last options (`use_xla`, `use_fp16` and `horovod`) are required for training on multi-node/multi-gpu settings
