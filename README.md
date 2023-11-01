# Training a multi-class classifier for increasing the accuracy of dictionary-based Named Entity Recognition

Scripts have been modified from the [NVIDIA codebase](https://github.com/NVIDIA/DeepLearningExamples) and adapted to work on the Finnish supercomputer [Puhti](https://docs.csc.fi/computing/systems-puhti/)

## Datasets
The consensus datasets that are used to train the model are available [here](https://doi.org/10.5281/zenodo.10008720).
There are 2 datasets included in the Zenodo project: 
1. A **small dataset** with ca.125,000 training and 62,500 development examples used to perform a grid search to detect the best set of hyperparameters (125k-w100_grid_search_set)
2. A **large dataset** of 12.5 million training and 62,500 testing examples to train the model used for prediction with the set of best hyperparameters identified above (12.5M-w100_train_test_set)


## Minimal installation instructions on a linux system

Clone the repository

```
git clone https://github.com/katnastou/BioBERT-based-entity-type-classifier.git
cd BioBERT-based-entity-type-classifier 
```

Download BioBERT base model

```
wget http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_v1.1_pubmed.tar.gz
mkdir -p models
tar -xvzf biobert_v1.1_pubmed.tar.gz -C models
rm biobert_v1.1_pubmed.tar.gz
```

Download training data from Zenodo

```
wget https://zenodo.org/api/records/10008720/files/125k-w100_grid_search_set.tar.gz
mkdir -p data
tar -xvzf 125k-w100_grid_search_set.tar.gz -C data
rm 125k-w100_grid_search_set.tar.gz
```

Install conda before proceeding. Instructions can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
If you are on a server with conda environments pre-installed, you could alternatively load one that supports at least Python 3.8. See detailed requirements for the nvidia-tensorflow package [here](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-wheel-release-notes/tf-wheel-rel.html#rel_23-03)

If you need to set up Python:

```
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
tar -xzvf Python-3.8.12.tgz
cd Python-3.8.12
./configure --prefix=$HOME/python38
make
make install
export PATH=$HOME/python38/bin:$PATH
#verify installation
python --version
```

```
python3.8 -m venv venv
source venv/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools
python3.8 -m pip install wheel
python3.8 -m pip cache purge
python3.8 -m pip install nvidia-pyindex==1.0.5 #or from file: https://files.pythonhosted.org/packages/64/4c/dd413559179536b9b7247f15bf968f7e52b5f8c1d2183ceb3d5ea9284776/nvidia-pyindex-1.0.5.tar.gz
python3.8 -m pip install nvidia-tensorflow[horovod]==1.15.5 #or from file: https://github.com/NVIDIA/tensorflow/archive/refs/tags/v1.15.5+nv23.03.tar.gz
```

And you are good to go!

Test your installation by running this script:

```
./run-entity-classification.sh
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
The command on a supercomputer with a slurm workload manager to rerun the training is: `sbatch slurm/slurm-run-finetuning-big.sh models/biobert_v1.1_pubmed 12.5M-w100_train_test_set 256 32 2e-5 1 consensus models/biobert_v1.1_pubmed/model.ckpt-1000000 data/biobert/other`
The results on the test set are: mean F-score=96.67% (SD=0).
This model has been used to run predictions and generate blocklists for all entity classes for [STRING database v12](https://string-db.org/), [DISEASES](https://diseases.jensenlab.org/Search) and [ORGANISMS](https://organisms.jensenlab.org/Search), as well as updating [dictionary files](https://jensenlab.org/resources/textmining/#dictionaries) for Jensenlab tagger. 

### Running on Prediction mode for large-scale runs

The script to run on prediction mode for all types is `run_predict_batch_auto_all_types.sh`. 
Running the bash script `./run_predict_batch_auto_che.sh` invokes the slurm script `slurm/slurm-run-predict.sh` to submit all jobs and generate predictions for all types, which are later used to generate probabilities.

Look at the `README` file and the setup scripts (`setup.sh` and `setup-general.sh`) within the `generate_prediction_inputs` directory for more details on how to run the entire pipeline from start to finish.



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

export PATH=/path/to/install_dir_for_openmpi/bin:$PATH
export LD_LIBRARY_PATH=/path/to/install_dir_for_openmpi/lib:$LD_LIBRARY_PATH
```

Instructions to install Tensorflow 1.15 with horovod support locally can be found here: https://www.pugetsystems.com/labs/hpc/how-to-install-tensorflow-1-15-for-nvidia-rtx30-gpus-without-docker-or-cuda-install-2005/


In order to train the method for span classification one needs to execute 
```
./run-entity-classification.sh
```

the shell script calls the `run_ner_consensus.py` with some default values. Check the script for more details. 


When using XLA on Puhti you might come across errors complaining about a file called `libdevice` or `ptxas`. These errors are caused by Puhti's environment being slightly broken with regards to CUDA, possibly due to inflexibility on CUDA's side. The problem can be solved by creating a symlink to the files in the BERT directory. Currently the files are in `/appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/bin/ptxas` and `/appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/nvvm/libdevice/libdevice.10.bc`, however these locations may change with CUDA updates. 


