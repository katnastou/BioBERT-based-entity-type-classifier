# Training a multi-class classifier for increasing the accuracy of dictionary-based Named Entity Recognition

Scripts have been modified from the [NVIDIA codebase](https://github.com/NVIDIA/DeepLearningExamples) and adapted to work on the Finnish supercomputer [Puhti](https://docs.csc.fi/computing/systems-puhti/)

## Steps to train/finetune the BERT model
### Datasets to train/finetune the model
The consensus datasets that are used to train the model are available [here](https://doi.org/10.5281/zenodo.10008720).
There are 2 datasets included in the Zenodo project: 
1. A dataset with ca.250,000 (ca.125,000 training and 125,000 development) examples used to perform a grid search to detect the best set of hyperparameters
2. A dataset of 12.5 million (10 million training, 62,500 development, 62,500 testing) examples to train the model used for prediction with the set of best hyperparameters identified above. 

### Grid search to find a set of hyperparameters for the big model
Run the script ./run-finetuning-grid.sh which invokes the script `slurm/slurm-run-finetuning-grid.sh` to run a grid search on the dataset with 125k examples with the following hyperparameters:
```
models = BioBERT base, BioBERT large
mini batch size = 32, 64
learning rate = 5e-5, 3e-5, 2e-5, 1e-5, 5e-6
number of epochs = 2, 3, 4
maximum sequence length = 96 (BioBERT large), 256 (BioBERT base)
```
Preliminary experiments showed that the maximum sequence length that could fit in memory provided the best results for the models and we decided to go with that option.

[Link with results from grid search](https://docs.google.com/spreadsheets/d/1YnDUO12wSxcg-MAqJ9_dg35hUwszVQRqeRPn6K76-MU/edit?usp=sharing)

The best model is BioBERT base with a `learning rate=2e-5,	number_of_epochs=4,	mini_batch_size=32,	MSL=256`, with a mean `F-score=94.93 (SD=0.017)` on dev set.

### Training the big model

### Running on Prediction mode for large scale runs

## Technical considerations on Puhti

In order for this to work one needs to have a working installation of Tensorflow 1.15 with horovod support. [Tensorflow 1.x support was deprecated on Puhti](https://docs.csc.fi/apps/tensorflow/), so one needs to set up an environment first before running the scripts.

```
module purge
#https://docs.csc.fi/computing/containers/tykky/
module load tykky
conda-containerize new --prefix conda-env env.yml
export PATH="path/to/conda-env/bin:$PATH"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install nvidia-pyindex==1.0.5
python -m pip install nvidia-tensorflow[horovod]==1.15.5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/venv/lib/
```

Instructions to install Tensorflow 1.15 with horovod support locally can be found here: https://www.pugetsystems.com/labs/hpc/how-to-install-tensorflow-1-15-for-nvidia-rtx30-gpus-without-docker-or-cuda-install-2005/


In order to train the method for span classification one needs to call the `run_ner_consensus.py` with the following commands:

```
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
```

where `replace_span` is the span to replace the named entity in focus, in case of masking (no replacement is done if left undefined), `task_name` is the name of the NER task (consensus in this case), `init_checkpoint` is the checkpoint of the model (should change to the trained model during prediction), `vocab_file` is the vocabulary file of BERT (same directory as the original model), `bert_config_file` is the configuration json of the model, `data_dir` is the directory with input data, `output_dir` is the directory with output data, `eval_batch_size` and `predict_batch_size` is the batch size, `max_seq_length` is the maximum sequence length in tokens, `learning_rate`, `num_training_epochs` are the initial learning rate and the maximum number of training epochs, `cased` is the flag to define whether to run with lower case or not, based on the model, `labels_dir` is the directory with the labels file containing the labels for the supervised training task and finally, the last options (`use_xla`, `use_fp16` and `horovod` are required for training on multi-node/multi-gpu settings). 

Example data can be found under `data/biobert/example-data`  and the labels file is under `data/biobert/other`

When using XLA on Puhti you might come across errors complaining about a file called `libdevice` or `ptxas`. These errors are caused by Puhti's environment being slightly broken with regards to CUDA, possibly due to inflexibility on CUDA's side. The problem can be solved by creating a symlink to the files in the BERT directory. Currently the files are in `/appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/bin/ptxas` and `/appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/nvvm/libdevice/libdevice.10.bc`, however these locations may change with CUDA updates. 
