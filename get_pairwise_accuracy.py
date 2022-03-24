# A simple way to compare the performance of vanilla elastic net vs wenda
# in predicting TP53 mutation across all pairwise TCGA sets (using one cancer
# type as training, another as source). Wenda generates models across varying
# values of k (increasing penalty on genes with low transferability to target
# distribution), so we're selecting the best value of k for a given pair.

# We're trying out several kinds of performance metrics, which can be specified
# using args.metric. Current options supported are:
#   accuracy (TP + TN / total number of samples)
#   auroc (ROC_AUC, area under curve of TPR vs FPR)
#   aupr (average precision, TP / TP + FP)
#   brier (a combination of accuracy and calibration)

import os
import re
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss 


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metric', default="accuracy", help="Metric used to quantify classifier performance.")
args = parser.parse_args()


kwnet_values = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 25, 35]

sources = []
targets = []
shuffled_or_signal = []
failures = []
performance_comp = []
wenda_performance = []
vanilla_performance = []


# Get all pairwise samples
dirs = os.listdir("data/pairwise")

for d in dirs:
    print(d)
    name = d.split("_")
    sources.append(name[1])
    targets.append(name[2])

    if 'shuffled' in os.path.basename(d):
        shuffled = True
    else:
        shuffled = False
    shuffled_or_signal.append(shuffled)

    # Pull actual mutation status from data_dir
    actual = pd.read_csv("data/pairwise/%s/target_y.tsv" % d, sep="\t", header=None)
    actual = np.asfortranarray(actual)

    # Find highest accuracy across values of k
    elastic_accuracy = 0
    best_wenda_accuracy = -1 if args.metric == "brier" else 0
    vanilla_trained = False
    some_wenda_trained = False

    for k in kwnet_values:
        # For some very high values of k, the models for a given pair did not train
        # so loading the text will throw an error. Given that we're interested in the
        # value of k with the highest accuracy, and it doesn't appear that any pairs
        # had no models that trained, here we can ignore the models that didn't train.
        output_dir = "output/pairwise/%s_wenda_gpu/k_{0:02d}".format(k) % d
        try:
            predicted = np.loadtxt(os.path.join(output_dir, "target_predictions.txt"))
            probs = np.loadtxt(os.path.join(output_dir, "target_probabilities.txt"))
            probs = probs[:, 0]
        except IOError:
            continue

        # Check if the model is just predicting all 0s or all 1s
        if predicted.sum() > 0 and predicted.sum() < predicted.size:
            if k == 0:
                vanilla_trained = True
            else:
                some_wenda_trained = True

        # Get accuracy or related metric
        if args.metric == "accuracy":
            correct = (predicted == actual)
            accuracy = correct.sum() / correct.size
        elif args.metric == "auroc":
            accuracy = roc_auc_score(actual, probs)
        elif args.metric == "aupr":
            accuracy = average_precision_score(actual, probs)
        elif args.metric == "brier":
            accuracy = (brier_score_loss(actual, probs)) * -1

        if k == 0:
            elastic_accuracy = accuracy
        elif accuracy > best_wenda_accuracy:
            best_wenda_accuracy = accuracy

    if not vanilla_trained and not some_wenda_trained:
        failures.append("both")
    elif not vanilla_trained:
        failures.append("wenda")
    elif not some_wenda_trained:
        failures.append("elastic")
    else:
        failures.append("neither")
    
    vanilla_performance.append(elastic_accuracy)
    performance_comp.append(best_wenda_accuracy - elastic_accuracy)
    wenda_performance.append(best_wenda_accuracy)
   
print(Counter(failures))
print(Counter(shuffled_or_signal))
print(performance_comp)

# If using brier score, flip back scores from negative
if args.metric == "brier":
    vanilla_performance = [i * -1 for i in vanilla_performance]
    wenda_performance = [i * -1 for i in wenda_performance]
    performance_comp = [i * -1 for i in performance_comp]

# Save to file
data = {'Source': sources, 'Target': targets, 'Failed': failures, 'Shuffled': shuffled_or_signal, 'Elastic': vanilla_performance, 'Wenda': wenda_performance, 'Comparative': performance_comp}
df = pd.DataFrame(data)
filename = "output/pairwise/pairwise_%s.tsv" % args.metric
df.to_csv(filename, index=False, sep="\t")
