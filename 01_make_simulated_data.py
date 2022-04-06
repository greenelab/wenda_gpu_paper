# Running wenda_orig on the full methylation dataset is intractable for us,
# so to do a direct comparison of the speed of training feature models with
# wenda_orig and wenda_gpu, we need smaller datasets. This borrows code from
# https://github.com/greenelab/simulate-groups, which generates datasets with
# some shared covariance structure.

# Params chosen:
# uncorr_frac=0.2 represents features (genes) that are not expressed and/or
#   irrelevant in a given biological context, here simulated to be 20% of features
# num_groups=10, number of interconnected gene "networks", arbitrarily set to 10
# group_sparsity=0.6 or 0.4, only some of the gene networks will affect the output
#   feature y, other gene networks have effect sizes set to 0. The decreased value
#   in the target dataset represents that two different datasets will have some
#   shared structure and some that isn't preserved between the two. Wenda will
#   ideally upweight genes with shared covariance structure between the datasets.

import os
import random
import numpy as np
from simulate_groups import simulate_ll

sizes = [100, 200, 500, 1000, 1500, 2000, 5000]
replicates = 3

# Get list of random seeds to pass to simulate_ll. If the seed is the same every
# time, you'll get a lot of repeat data, which may cause problems for training.
random.seed(1387)
seeds=random.sample(range(0,10000), len(sizes)*replicates*2)

for size in sizes:
    for i in range(replicates):
        prefix = "sim_%d_rep_%d" % (size, i)
        data_dir = "data/%s" % prefix
        os.makedirs(data_dir, exist_ok = True)

        # Generate source data
        source_x, source_y, info_dict = simulate_ll(n=1200, p=size, uncorr_frac=.2,
                num_groups=10, group_sparsity=0.6, seed=seeds.pop())
        np.savetxt(os.path.join(data_dir, "source_data.tsv"), source_x,
                delimiter="\t", fmt="%.4f")
        np.savetxt(os.path.join(data_dir, "source_y.tsv"), source_y, fmt="%i")
        
        # Generate target data with some variable effect sizes from source preserved
        source_betas = info_dict["betas"]
        target_x, target_y, info_dict = simulate_ll(n=1000, p=size, uncorr_frac=.2,
                num_groups=10, group_sparsity=0.4, seed=seeds.pop(),
                prev_betas=source_betas, prev_groups=info_dict['groups_to_keep'])
        np.savetxt(os.path.join(data_dir, "target_data.tsv"), target_x,
                delimiter="\t", fmt="%.4f")
        np.savetxt(os.path.join(data_dir, "target_y.tsv"), target_y, fmt="%i")
