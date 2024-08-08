# browDataAug

This repository contains the code used for the paper "Brownian motion data augmentation: a method to push neural network performance on nanopore sensors" by Javier Kipen and Joakim Jaldén. 

## Setup

The first step is to clone the QuipuNet repository (https://github.com/kmisiunas/QuipuNet) in the "ext" folder. Then, run the Python script scripts/Common/DatasetFuncs.py, which preprocesses the dataset as done in Quipunet and saves it in an hdf5 file within the data/ folder. This step is not done before because the dataset would consume a considerable amount of space in the repo.

For the proof of the increased accuracy with Brownian Augmentation on QuipuNet, the augmentation method was implemented with a custom C++ routine, which was then bound to Tensorflow and is located in the ext/ folder. This routine needs to be compiled following the instructions in the folder. Once compiled, the Python virtual environment used is specified in "containers/BrownianAccImprov_requirements.txt." Note that Tensorflow 2.11 CUDA 11.7 was used.


For the training of YupanaNet, we implemented all other augmentations (noise addition, magnitude multiplication, uniform stretching) and the Brownian augmentation in a cupy routine. This reduced the time spent augmenting considerably, but it had to be done in another environment. This part was done with the container specified in "containers/YupanaNetContainer.def" which uses Tensorflow 2.14. 

Since the optimized augmentation could be helpful for other people in the field, a new repository was created to update and optimize this tool to augment nanopore reads: https://github.com/JavierKipen/NanoporeDataAugmenter. 


## Overview of repo structure

### Overview of folders

- Containers: Has the files to reproduce the environments to run the scripts.
- Data: Contains the dataset used(has to be generated according to the setup instructions).
- Ext: Has the external code needed.
- Results: Have all the results used for the paper, primarily results after training of different neural networks.
- Scripts: Has the analysis of the results and the scripts to generate the results.

### Script and results subfolders

- Common: Has useful code for other subfolders, such as dataset loading, training, model generation, and common parameters.
- Figures_paper: Code used to generate figures and calculations in the Supplementary Information
- NewNN: Has the tuning of YupanaNet, some experiments on architectures, and the final results of the newer neural network.
- Reproduction_Quipu: Reproduces the results of QuipuNet, first as the training showed in their notebook, and then we optimize the training.
- Validation_Brow_Aug: Code to validate the increase in accuracy due to the Brownian Augmentation.




