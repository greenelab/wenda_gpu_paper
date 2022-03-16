import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

from statistics import median
from scipy.stats import pearsonr

from plotnine import ggplot, geom_point, aes, xlab, ylab, geom_tile, theme_bw
from plotnine import scale_color_brewer, geom_line, geom_abline, geom_pointrange

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kval", default=6, help="k value to use for methylation data.")
args = parser.parse_args()

# Figure 1A
# Speed comparisons

# Get all files with simulation runtime information
runtime_files = glob.glob("runtimes/simulated/sim_*", recursive=False)

# Parse into dataframe
runtimes = pd.DataFrame(columns=['Features','Software','Replicate','Time'])
for f in runtime_files:
    filename=f.split("_")
    size=int(filename[1])
    rep=filename[3]
    platform=re.sub(".txt","",filename[5])
    content=open(f,"r").read()
    time = int(content.strip().split(": ")[1])
    row=pd.DataFrame(
            [[size, platform, rep, time]],
            columns=['Features','Software','Replicate','Time'])
    runtimes = runtimes.append(row)

# Get mean, min, and max runtime for each set of replicates
runtimes['Features']=pd.to_numeric(runtimes.Features)
runtimes['Time']=pd.to_numeric(runtimes.Time)
g = runtimes.groupby(['Features','Software']).agg({'Time': ['mean','min','max']}).reset_index()

# Just gonna redo it and work on the rest later I'm so tired
new_runtimes = pd.DataFrame({
    'Features': g['Features'],
    'Software': g['Software'],
    'Mean': g['Time']['mean'],
    'Min': g['Time']['min'],
    'Max': g['Time']['max']})

print(new_runtimes)

# Plot runtimes
gA = ggplot(new_runtimes) + geom_pointrange(aes('Features', 'Mean', ymin='Min', ymax='Max'))
gA += geom_line(aes('Features', 'Mean', group='Software'))
gA += ylab("Time (seconds)") + theme_bw()
gA.save("figures/runtimes.png", dpi=300)

# Figure 1B
# Comparing age predictions from wenda_orig and wenda_gpu

# Load wenda_gpu elastic net predictions
output_dir = "output/handl_wenda_gpu/k_{0:02d}".format(args.kval)
gpu_prediction_path = os.path.join(output_dir, "target_predictions.txt")
gpu_predictions = np.loadtxt(gpu_prediction_path)

# Load wenda_orig elastic net predictions
# wenda_orig does predictions over 10-fold cross validation, so we will get the correlation
# of our data against all 10 values and use the set closest to the median for the plot
output_dir = "output/handl_wenda_orig/wnet_cv/"

orig_predictions = np.zeros((1001, 10))
correlations = []
for i in range(10):
    repeat_dir = os.path.join(output_dir, "repetition_{0:02d}".format(i))
    orig_prediction_path = os.path.join(repeat_dir, "predictions_k%d.txt" % args.kval)
    o_pred = np.loadtxt(orig_prediction_path)
    orig_predictions[:,i] = o_pred
    correlations.append(pearsonr(o_pred, gpu_predictions)[0])

# Get set with median correlation
correlations = np.asarray(correlations)
idx = (np.abs(correlations - median(correlations))).argmin()
median_set = orig_predictions[:,idx]

# Plot correlation
data = pd.DataFrame(
        {'gpu': gpu_predictions,
         'orig': median_set})

gB = ggplot(data, aes('gpu', 'orig')) + geom_point()
gB += xlab("wenda_gpu predicted age") + ylab("wenda_orig predicted age")
gB += theme_bw()
gB.save("figures/software_correlation.png", dpi=300)


# Figure 1C
# Comparing age predictions from wenda_gpu and vanilla elastic net

# Load phenotype data with actual age and tissue type
pheno_data_path = "data/handl/target_phenotypes.tsv"
pheno_data = pd.read_csv(pheno_data_path, delimiter="\t")
true_ages = np.asarray(pheno_data['age'])

# Load vanilla elastic net predictions
output_dir = "output/handl_wenda_gpu/k_00"
vanilla_prediction_path = os.path.join(output_dir, "target_predictions.txt")
vanilla_predictions = np.loadtxt(vanilla_prediction_path)

# Plot age results
data = pd.DataFrame(
        {'gpu': gpu_predictions,
        'vanilla': vanilla_predictions,
        'true': true_ages,
        'tissue': pheno_data['paper_tissue']})

gC = ggplot(data, aes('true', 'gpu', color='tissue')) + geom_point()
gC += ylab("wenda_gpu predicted age") + xlab("Actual age")
gC += scale_color_brewer(type="qual", palette="Set1")
gC += geom_abline() + theme_bw()
gC.save("figures/wenda_true_comparison.png", dpi=300)

gCC = ggplot(data, aes('true', 'vanilla', color='tissue')) + geom_point()
gCC += ylab("Elastic net predicted age") + xlab("Actual age")
gCC += scale_color_brewer(type="qual", palette="Set1")
gCC += geom_abline() + theme_bw()
gCC.save("figures/vanilla_true_comparison.png", dpi=300)
