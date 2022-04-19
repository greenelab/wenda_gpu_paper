# wenda_gpu_paper

This repository has all the scripts run to perform the analysis in the wenda_gpu paper (preprint forthcoming). 
For the software package that this repo analyzes, go here: https://github.com/greenelab/wenda_gpu.

The goal of the paper is to demonstrate that our new implementation of weighted elastic net domain adaptation, called wenda_gpu, is significantly faster than the original CPU method while returning comparable results.

![Age prediction with elastic net](https://github.com/greenelab/wenda_gpu_paper/blob/main/figures/vanilla_true_comparison.png)
![Age prediction with wenda](https://github.com/greenelab/wenda_gpu_paper/blob/main/figures/wenda_true_comparison.png)


## Setup

All processing and analysis scripts were performed using the conda environment specified in environment.yaml. 
To build and activate this environment run:
```
conda env create --file environment.yaml
conda activate wenda_gpu_paper
```

## Data

This repository contains the data for the predicting age via tissue methylation analysis, first used in the original wenda paper (Handl et al. 2019, DOI 10.1093/bioinformatics/btz338).
All other data can be generated or downloaded using scripts. 
The simulated datasets for runtime comparison are generated using 01_make_simulated_data.py.
The TCGA data for pairwise mutation prediction is downloaded using 05_download_tcga_data.py.


## Usage

All analyses can be run by executing the numbered scripts.
Some of these are performing analyses in their own right (end in .py), and some of them are master scripts that simply execute some number of other scripts (end in .sh). 
Some of these master scripts call a large number of scripts and can take up to several days to run, which will be indicated in the master script's header.
These scripts can be safely broken into chunks at user discretion.
For instance, users with access to large cluster resources may want to break them into chunks and run them in parallel.
