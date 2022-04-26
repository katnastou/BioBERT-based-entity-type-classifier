# Training a multi-class classifier for increasing the accuracy of dictionary-based Named Entity Recognition

Scripts have been modified from the [NVIDIA codebase](https://github.com/NVIDIA/DeepLearningExamples) and adapted to work on the Finnish supercomputer [Puhti](https://docs.csc.fi/computing/systems-puhti/)

## Steps to finetune the BERT model
### Datasets to train the model
The consensus datasets that are used to train the model are available [here]().
There are 2 datasets generated: a 125,000 training example dataset that is used to perform a grid search to detect the best set of hyperparameters to finetune the model on a big dataset of 12.5 million training examples. 

### Grid search to find a set of hyperparameters for the big model
Run the script ./run-finetuning-grid.sh which invokes the script slurm/slurm-run-finetuning-grid.sh to run a grid search on the dataset with 125k examples with the following hyperparameters:
models = BioBERT base, BioBERT large
mini batch size = 32, 64
learning rate = 5e-5, 3e-5, 2e-5, 1e-5, 5e-6
number of epochs = 2, 3, 4
maximum sequence length = 96 (BioBERT large), 256 (BioBERT base)
Preliminary experiments showed that the maximum sequence length that could fit in memory provided the best results for the models and we decided to go with that option.

## Technical considerations on Puhti

When using XLA on Puhti you might come across errors complaining about a file called libdevice or ptxas. These errors are caused by Puhti's environment being slightly broken with regards to CUDA, possibly due to inflexibility on CUDA's side. The problem can be solved by creating a symlink to the files in the BERT directory. Currently the files are in /appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/bin/ptxas and /appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/nvvm/libdevice/libdevice.10.bc, however these locations may change with CUDA updates. 