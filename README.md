# Training a multi-class classifier for increasing the accuracy of dictionary-based Named Entity Recognition

Scripts have been modified from the [NVIDIA codebase](https://github.com/NVIDIA/DeepLearningExamples) and adapted to work on the Finnish supercomputer [Puhti](https://docs.csc.fi/computing/systems-puhti/)

## Datasets
The consensus datasets that are used to train the model are available [here](https://doi.org/10.5281/zenodo.10008720).
There are 2 datasets included in the Zenodo project: 
1. A **small dataset** with ca.125,000 training and 62,500 development examples used to perform a grid search to detect the best set of hyperparameters (125k-w100_grid_search_set)
2. A **large dataset** of 10 million training and 62,500 testing examples to train the model used for prediction with the set of best hyperparameters identified above (12.5M-w100_train_test_set)


## Minimal installation instructions on a linux system
clone the repository

```
git clone https://github.com/katnastou/BioBERT-based-entity-type-classifier.git
cd BioBERT-based-entity-type-classifier 
```

Download BioBERT model

```wget http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_v1.1_pubmed.tar.gz
tar -xvzf biobert_v1.1_pubmed.tar.gz -C models
rm biobert_v1.1_pubmed.tar.gz
```

Download training data

```wget https://zenodo.org/api/records/10008720/draft/files/125k-w100_grid_search_set.tar.gz #update link 
tar -xvzf 125k-w100_grid_search_set.tar.gz
rm 125k-w100_grid_search_set.tar.gz
```
```
conda update -n base -c conda-forge conda
conda env create --name conda-env -f env.yml
conda activate conda-env
conda update pip
brew install libuv
pip install --user -r requirements.txt
HOROVOD_WITHOUT_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir --user horovod

docker pull nvcr.io/nvidia/tensorflow:23.02-tf1-py3

./run_entity_classification.py


cd "/usr/local/Cellar/bazel/6.4.0/libexec/bin" && curl -fLO https://releases.bazel.build/0.25.3/release/bazel-0.25.3-darwin-x86_64 && chmod +x bazel-0.25.3-darwin-x86_64
cd
conda update -n base -c conda-forge conda
conda env create --name conda-env -f env.yml
conda activate conda-env
wget https://github.com/NVIDIA/tensorflow/archive/refs/tags/v1.15.5+nv23.03.zip
unzip v1.15.5+nv23.03.zip
rm v1.15.5+nv23.03.zip
cd tensorflow-1.15.5-nv23.03
./configure
bazel build //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.15.5-nv23.03-*.whl
```

## Steps to train/finetune the model on the Puhti supercomputer


### Grid search to find a set of hyperparameters 
Run the script `./run-finetuning-grid.sh` which invokes the script `slurm/slurm-run-finetuning-grid.sh` to run a grid search with the small dataset with the following hyperparameters:
```
models = BioBERT base
mini batch size = 32, 64
learning rate = 5e-5, 3e-5, 2e-5, 1e-5, 5e-6
number of epochs = 2, 3, 4
maximum sequence length = 256
repetitions = 3
```
Preliminary experiments showed that the maximum sequence length that could fit in memory provided the best results for the models and we decided to go with that option.

[Link with results from grid search](https://docs.google.com/spreadsheets/d/1kfypTjb_1YUncyqHSgwaD2fEjNaxGCF9vMigj87tB9E/edit?usp=sharing)

The hyperparameters for the best-performing model on the development set are `learning rate=2e-5,	number_of_epochs=2,	mini_batch_size=32,	MSL=256`, with a mean `F-score=94.83% (SD=0.0356)`.

To get the stats for the finetuning grid run: `python3 get_stat.py <logs_dir> <output_filename>` (e.g. `python3 get_stat.py finetuning-grid-logs/ finetuning-grid-results.tsv`)

### Training the model with the large dataset

We have trained a model with the large dataset using the best set of hyperparameters. 
The command on a slurm supercomputer to rerun the training is: `sbatch slurm/slurm-run-finetuning-big.sh models/biobert_v1.1_pubmed 12.5M-w100_train_test_set 256 32 2e-5 1 consensus models/biobert_v1.1_pubmed/model.ckpt-1000000 data/biobert/other`
The results on the test set are: mean F-score= (SD=).
This model has been used to run predictions and generate blocklists for all entity classes for [Jensenlab resources](https://jensenlab.org/resources/). 

### Running on Prediction mode for large scale runs

The scripts to run on prediction mode are `run_predict_batch_auto_{che,org,dis,ggp}.sh`. 
If, for example, one runs the script `./run_predict_batch_auto_che.sh` the script `slurm/slurm-run-predict.sh` will be invoked and run predictions on all examples of chemical mentions.



## Technical considerations on Puhti

In order for this to work one needs to have a working installation of Tensorflow 1.15 with horovod support. [Tensorflow 1.x support was deprecated on Puhti](https://docs.csc.fi/apps/tensorflow/), so one needs to set up an environment first before running the scripts.

```
module purge
#https://docs.csc.fi/computing/containers/tykky/
module load tykky
conda-containerize new --prefix conda-env env.yml

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install nvidia-pyindex==1.0.5
python -m pip install nvidia-tensorflow[horovod]==1.15.5

#set up openmpi version 4.0.1
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar -xzf openmpi-4.0.1.tar.gz
rm openmpi-4.0.1.tar.gz 
cd openmpi-4.0.1
./configure --prefix=/path/to/install_dir_for_openmpi
make all
make install

export PATH=/users/katenast/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/users/katenast/openmpi/lib:$LD_LIBRARY_PATH
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


