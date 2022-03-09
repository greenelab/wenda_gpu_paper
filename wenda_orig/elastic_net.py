# This code is from https://github.com/PfeiferLabTue/wenda, adapted from
# main_real_data.py. The script skips over training models and computing
# confidences, which we got from the original authors, and only runs the
# elastic net step. Originally written by Lisa (Handl) Eisenberg


import os
import sys
sys.path.append("")

import pandas
import numpy as np
import sklearn.linear_model as lm
import feature_models
import GPy
import itertools
import pandas as pd

import time
import datetime

import data as data_mod
import models
import util


def printTestErrors(pred_raw, test_y_raw, heading=None, indent=0):
    prefix = " "*indent
    errors = np.abs(test_y_raw - pred_raw)
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    corr = np.corrcoef(pred_raw, test_y_raw)[0,1]
    std = np.std(errors)
    q75, q25 = np.percentile(errors, [75 ,25])
    iqr = q75 - q25
    if (heading is not None):
        print(prefix + heading)
        print(prefix + len(heading)*'-')
    print(prefix + "Mean abs. error:", mean_err)
    print(prefix + "Median abs. error:", median_err)
    print(prefix + "Correlation:", corr)
    print()
    return [mean_err, median_err, corr, std, iqr]


def main(prefix):
     
    # print time stamp
    print("Program started:", "{:%d.%m.%Y %H:%M:%S}".format(datetime.datetime.now()))
    print(flush=True)
     
    # set paths and parameters
    home_path = "."
      
    feature_model_path = os.path.join(home_path, "feature_models")
    conf_path = os.path.join(home_path, "confidences")
    output_path = os.path.join(home_path, "output")
        
    n_jobs = 10
    # power parameters for weight function
    kwnet_values = [1,2,3,4, 6, 8, 10, 14, 18, 25, 35]
    repetitions = 10
    
    print("feature_model_path:", feature_model_path)
    print("conf_path:", conf_path)
    print("output_path:", output_path)
    print("n_jobs:", n_jobs)
    print("weighting parameters:", kwnet_values)
    print("repetitions:", repetitions)
    print()
    
    print("Reading data...", end=" ", flush=True)
    start = time.time()
    
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
    output_path = os.path.join(output_path, output_prefix)

    # collect and aggregate tissue information
    pheno_table = pd.read_csv("data/handl/target_phenotypes.tsv", sep="\t")
    test_tissues = pheno_table["tissue_complete"]
    
    test_tissues_aggregated = test_tissues.copy()
    test_tissues_aggregated[test_tissues_aggregated == "whole blood"] = "blood"
    test_tissues_aggregated[test_tissues_aggregated == "menstrual blood"] = "blood"
    test_tissues_aggregated[test_tissues_aggregated == "Brain MedialFrontalCortex"] = "Brain Frontal"

    crbm_samples = np.array(test_tissues == "Brain CRBM")
    
    normalizer_x = util.StandardNormalizer
    normalizer_y = util.HorvathNormalizer

    feature_model_type = feature_models.FeatureGPR
    feature_model_params = {"kernel": GPy.kern.Linear(input_dim=data.training.getNofCpGs()-1)}

    model = models.Wenda(data.training.meth_matrix, data.training.age, data.test.meth_matrix, 
                                 normalizer_x, normalizer_y, feature_model_path, feature_model_type, feature_model_params, conf_path, 
                                 n_jobs=n_jobs)
    print(model)
    print(flush=True)
   
    # Predict with CV on training data ---
    print("\nPredicting with cross-validation on training data...")
       
    for i in range(repetitions):
        print("- repetition ", i, "\n", flush=True)
           
        # setup output path
        adaptive_cv_path = os.path.join(output_path, "wnet_cv/repetition_{0:02d}".format(i))
        os.makedirs(adaptive_cv_path, exist_ok=True)
        print("  path:", adaptive_cv_path, "\n", flush=True)
           
        # iterate over weighting functions, predict and save errors
        for k_wnet in kwnet_values:
            print("  - k_wnet =", k_wnet, end="\n    ", flush=True)
            weighting_function = lambda x: np.power(1-x, k_wnet)
            predictions = model.predictWithTrainingDataCV(weight_func=weighting_function, grouping=test_tissues_aggregated, alpha=0.8, n_splits=10,
                                                          predict_path=os.path.join(adaptive_cv_path, "glmwnet_pow{0:d}".format(k_wnet)))
            np.savetxt(os.path.join(adaptive_cv_path, "predictions_k{0:d}.txt".format(k_wnet)), predictions)
            
            errors_all = printTestErrors(predictions, data.test.age, "Weighted elastic net (full data):", indent=4)
            errors_crbm = printTestErrors(predictions[crbm_samples], data.test.age[crbm_samples], "Weighted elastic net (CRBM):", indent=4)
                      
            table = pandas.DataFrame(np.vstack([errors_all, errors_crbm]), 
                                     index=["all", "crbm"], 
                                     columns=["mean", "median", "corr", "std", "iqr"])
            table.to_csv(os.path.join(adaptive_cv_path, "errors_k{0:d}.csv".format(k_wnet)), sep=";", quotechar='"')
            
        np.savetxt(os.path.join(adaptive_cv_path, "powers.txt"), kwnet_values)

prefix=sys.argv[1]
main(prefix)
