# This code is from https://github.com/PfeiferLabTue/wenda, adapted from
# main_real_data.py. The script is truncated to only train feature models
# and compute confidences, without the final elastic net step that we won't
# use for runtime analysis. Originally written by Lisa (Handl) Eisenberg


import os
import sys
sys.path.append("")

import pandas
import numpy as np
import sklearn.linear_model as lm
import feature_models
import GPy
import itertools

import time
import datetime

import data as data_mod
import models
import util


def main(prefix):

    # print time stamp
    print("Program started:", "{:%d.%m.%Y %H:%M:%S}".format(datetime.datetime.now()))
    print(flush=True)

    # set paths and parameters
    home_path = "."

    feature_model_path = os.path.join(home_path, "feature_models")
    conf_path = os.path.join(home_path, "confidences")

    n_jobs = 10
    # power parameters for weight function
    kwnet_values = [1,2,3,4, 6, 8, 10, 14, 18, 25, 35]

    print("feature_model_path:", feature_model_path)
    print("conf_path:", conf_path)
    print("n_jobs:", n_jobs)
    print("weighting parameters:", kwnet_values)
    print()

    print("Reading data...", end=" ", flush=True)
    start = time.time()

    print(prefix)
    data = data_mod.DataDNAmethPreprocessed(prefix)

    end = time.time()
    print("took", util.sec2str(end-start))
    print()
    print("data:", data)
    print(flush=True)

    # add in prefix for output directories
    output_prefix = prefix + "_wenda_orig"
    feature_model_path = os.path.join(feature_model_path, output_prefix)
    conf_path = os.path.join(conf_path, output_prefix)
    print(feature_model_path)
    print(conf_path)

    normalizer_x = util.StandardNormalizer
    normalizer_y = util.NoNormalizer

    feature_model_type = feature_models.FeatureGPR
    feature_model_params = {"kernel": GPy.kern.Linear(input_dim=data.training.getNofCpGs()-1)}

    model = models.Wenda(data.training.meth_matrix, data.training.age, data.test.meth_matrix,
                                 normalizer_x, normalizer_y, feature_model_path, feature_model_type,
                                 feature_model_params, conf_path, n_jobs=n_jobs)
    print(model)
    print(flush=True)

    # fit feature models
    print("Fitting feature models...", flush=True)
    model.fitFeatureModels()
    print("Collecting confidences...", flush=True)
    model.collectConfidences()


prefix=sys.argv[1]
main(prefix)
