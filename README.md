# wenda_gpu_paper

This repository has all the scripts run to perform the analysis in the wenda_gpu paper (preprint forthcoming). 
For the reusable software, go here: https://github.com/greenelab/wenda_gpu.

The goal of the paper is to demonstrate that our new implementation of weighted elastic net domain adaptation, called wenda_gpu, is significantly faster than the original CPU method while returning comparable results.


## Data

This repository contains the data for the predicting age via tissue methylation analysis, first used in the original wenda paper (Handl et al. 2019, DOI 10.1093/bioinformatics/btz338).
All other data can be generated or downloaded using scripts. 
The simulated datasets for runtime comparison are generated using 01_make_simulated_data.py.
The TCGA data for pairwise mutation prediction is downloaded using 05_download_tcga_data.py.


## Pipeline

All analyses can be run by executing the numbered scripts.
Some of these are performing analyses in their own right (end in .py), and some of them are master scripts that simply execute some number of other scripts (end in .sh). 
Some of these master scripts call a large number of scripts and can take up to several days to run, which will be indicated in the master script's header.
These scripts can be safely broken into chunks at user discretion.
Users with access to large cluster resources may want to break them into chunks and run them in parallel.
