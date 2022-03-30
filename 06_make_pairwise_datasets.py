# Script that generates source + target datasets for pairs of TCGA cancer types.
# This is pulled directly from https://github.com/greenelab/pancancer-evaluation/
# run_cross_cancer_classification.py, but we're only using the script to make the
# datasets, not to train the classifier, so several components of the original
# code are removed. It's also designed to work for many genes and many cancer types,
# but since we're using this as a proof of principle we stuck to TP53 and hand-
# selected cancer types that had non-trivial numbers of both TP53 mutant and WT
# and hardcoded them in. Originally written by Jake Crawford.

import sys
import argparse
import itertools as it
from pathlib import Path

import numpy as np

import pandas as pd
from tqdm import tqdm

import pancancer_evaluation.config as cfg
from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
import pancancer_evaluation.utilities.data_utilities as du

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='use subset of data for fast debugging')
    p.add_argument('--results_dir', default="results",
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--subset_mad_genes', type=int, default=8000,
                   help='if included, subset gene features to this number of '
                        'features having highest mean absolute deviation')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    args.results_dir = Path(args.results_dir).resolve()

    return args

if __name__ == '__main__':

    # process command line arguments
    args = process_args()

    # create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tcga_data = TCGADataModel(seed=args.seed,
                              subset_mad_genes=args.subset_mad_genes,
                              verbose=args.verbose,
                              debug=args.debug)

    identifiers = ['TP53_GBM', 'TP53_LUAD', 'TP53_LUSC', 'TP53_UCEC', 'TP53_BLCA',
                   'TP53_ESCA', 'TP53_PAAD', 'TP53_LIHC', 'TP53_SARC', 'TP53_BRCA',
                   'TP53_COAD', 'TP53_STAD', 'TP53_SKCM', 'TP53_HNSC', 'TP53_READ', 'TP53_LGG']

    # create output directory
    output_dir = Path(args.results_dir, 'cross_cancer').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        outer_progress = tqdm(identifiers,
                              total=len(identifiers),
                              ncols=100,
                              file=sys.stdout)

        for train_identifier in outer_progress:
            outer_progress.set_description('train: {}'.format(train_identifier))

            try:
                train_classification = du.get_classification(
                    train_identifier.split('_')[0])
            
            except (KeyError, IndexError) as e:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias, TODO check for this later)
                print('Identifier not found in mutation data, skipping',
                      file=sys.stderr)
                continue

            inner_progress = tqdm(identifiers,
                                  total=len(identifiers),
                                  ncols=100,
                                  file=sys.stdout)

            for test_identifier in inner_progress:
            
                inner_progress.set_description('test: {}'.format(test_identifier))

                try:
                    test_classification = du.get_classification(
                        test_identifier.split('_')[0])
                    tcga_data.write_datasets_to_file(train_identifier,
                                                     test_identifier,
                                                     train_classification,
                                                     test_classification,
                                                     output_dir,
                                                     shuffle_labels)
                except (KeyError, IndexError) as e:
                    # this might happen if the given gene isn't in the mutation data
                    # (or has a different alias, TODO check for this later)
                    print('Identifier not found in mutation data, skipping',
                          file=sys.stderr)
                    continue
