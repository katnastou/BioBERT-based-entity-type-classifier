# Training a multi-class classifier for increasing the accuracy of dictionary-based Named Entity Recognition

Scripts have been modified from the [NVIDIA codebase](https://github.com/NVIDIA/DeepLearningExamples) and adapted to work on the Finnish supercomputer [Puhti](https://docs.csc.fi/computing/systems-puhti/)

## Steps to finetune the BERT model


## Technical considerations on Puhti

When using XLA on Puhti you might come across errors complaining about a file called libdevice or ptxas. These errors are caused by Puhti's environment being slightly broken with regards to CUDA, possibly due to inflexibility on CUDA's side. The problem can be solved by creating a symlink to the files in the BERT directory. Currently the files are in /appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/bin/ptxas and /appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/nvvm/libdevice/libdevice.10.bc, however these locations may change with CUDA updates. 