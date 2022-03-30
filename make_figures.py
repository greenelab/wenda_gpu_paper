# This script is used to generate all the figures used in both the poster and the paper.
# It makes separate files for each figure and also a combined one using svgutils.

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import svgutils.transform as sg

from lxml import etree
from statistics import median
from scipy.stats import pearsonr

from plotnine import ggplot, geom_point, aes, xlab, ylab, geom_tile, theme_bw
from plotnine import scale_color_brewer, geom_abline, geom_pointrange
from plotnine import scale_fill_gradient2, geom_line, theme, element_text
from plotnine import scale_color_manual, scale_fill_manual, labs, ggtitle
from plotnine import annotate, theme_classic, scale_x_discrete, geom_bar

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kval", default=6, help="k value to use for methylation data.")
parser.add_argument("-p", "--png", action="store_true", help="whether to save plots as SVGs or PNGs.")
args = parser.parse_args()

# File extension for saved images
if args.png:
    ext = "png"
else:
    ext = "svg"


# Figure 0
# Basic barplot of methylation runtimes, only for the AACR poster
# 183600: original wenda paper said theirs ran in 51 hours, so 51*60*60
# 246213: total time time_wenda_orig.sh was allowed to run on handl dataset
# 18566: taken from runtimes/handl_wenda_gpu.txt
times = pd.DataFrame({
    'Software': ['wenda_orig, 10 cores', 'wenda_orig, 6 cores', 'wenda_gpu'],
    'Time': [183600, 246213, 18566]})
times['Time'] = times['Time'] / 3600

g0 = ggplot(times, aes(x='Software', y='Time', fill='Software')) + geom_bar(stat='identity')
g0 += scale_x_discrete(limits=['wenda_orig, 10 cores', 'wenda_orig, 6 cores', 'wenda_gpu'])
g0 += scale_fill_manual(["#E41A1C", "#4DAF4A", "#377EB8"], guide=False)
g0 += ggtitle("Software Runtimes on Methylation Dataset (12980 features)")
g0 += theme_bw()
g0 += ylab("Time (Hours)")
g0 += annotate("text", x=2, y=70, label="*Did not finish")
g0.save("figures/methyl_runtimes.%s" % ext, dpi=300)


# Figure 1A
# Speed comparisons

# Get all files with simulation runtime information
runtime_files = glob.glob("runtimes/simulated/sim_*", recursive=False)

# Parse into dataframe
runtimes = pd.DataFrame(columns=['Features', 'Software', 'Replicate', 'Time'])
for f in runtime_files:
    filename = f.split("_")
    size = int(filename[1])
    rep = filename[3]
    platform = re.sub(".txt", "", filename[5])
    platform = "wenda_" + platform
    content = open(f, "r").read()
    time = int(content.strip().split(": ")[1])
    row = pd.DataFrame(
            [[size, platform, rep, time]],
            columns=['Features', 'Software', 'Replicate', 'Time'])
    runtimes = runtimes.append(row)

# Get mean, min, and max runtime for each set of replicates
runtimes['Features'] = pd.to_numeric(runtimes.Features)
runtimes['Time'] = pd.to_numeric(runtimes.Time)
g = runtimes.groupby(['Features', 'Software']).agg({'Time': ['mean', 'min', 'max']}).reset_index()

# Lazy workaround because reindexing the dataframe wasn't working
new_runtimes = pd.DataFrame({
    'Features': g['Features'],
    'Software': g['Software'],
    'Mean': g['Time']['mean'],
    'Min': g['Time']['min'],
    'Max': g['Time']['max']})

# Plot runtimes
gA = ggplot(new_runtimes) + geom_pointrange(aes('Features', 'Mean', ymin='Min', ymax='Max', color='Software'))
gA += geom_line(aes('Features', 'Mean', group='Software', color='Software'))
gA += scale_color_brewer(type='qual', palette='Set1')
gA += theme_bw()
gA += ylab("Time (seconds)")
gA += ggtitle("Software Runtimes of Simulated Datasets")
gA.save("figures/runtimes.%s" % ext, dpi=300)


# Figure 1B
# Comparing age predictions from wenda_orig and wenda_gpu

# Load wenda_gpu elastic net predictions
output_dir = "output/handl_wenda_gpu/k_{0:02d}".format(args.kval)
gpu_prediction_path = os.path.join(output_dir, "target_predictions.txt")
gpu_predictions = np.loadtxt(gpu_prediction_path)

# Load wenda_orig elastic net predictions
# wenda_orig does predictions over 10-fold cross validation, so we will get the
# correlation of our data against all 10 values and use the set closest to the
# median for the plot
output_dir = "output/handl_wenda_orig/wnet_cv/"

orig_predictions = np.zeros((1001, 10))
correlations = []
for i in range(10):
    repeat_dir = os.path.join(output_dir, "repetition_%0.02d" % i)
    orig_prediction_path = os.path.join(repeat_dir, "predictions_k%d.txt" % args.kval)
    o_pred = np.loadtxt(orig_prediction_path)
    orig_predictions[:, i] = o_pred
    correlations.append(pearsonr(o_pred, gpu_predictions)[0])

# Get CV run with closest to median correlation
correlations = np.asarray(correlations)
idx = (np.abs(correlations - median(correlations))).argmin()
median_set = orig_predictions[:, idx]

# Plot correlation
data = pd.DataFrame(
        {'gpu': gpu_predictions,
         'orig': median_set})

gB = ggplot(data, aes('gpu', 'orig')) + geom_point(size=2)
gB += xlab("wenda_gpu predicted age") + ylab("wenda_orig predicted age")
gB += theme_bw()
gB += ggtitle("Methylation Age Predictions Across Softwares")
gB.save("figures/software_correlation.%s" % ext, dpi=300)


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
data['in_source'] = np.where(data['tissue'] == "Brain Cerebellum", "2", "1")

gC = ggplot(data, aes('true', 'gpu', color='tissue'))
gC += geom_abline(color='gray', linetype='dashed')
gC += geom_point(size=2)
gC += labs(color='Tissue', x='Actual age', y='Predicted age')
gC += scale_color_manual(["#377EB8", "#E41A1C", "#A65628",
                          "#F781BF", "#4DAF4A", "#984EA3"])
gC += theme_bw()
gC += ggtitle("Methylation Age Prediction with wenda_gpu")
gC.save("figures/wenda_true_comparison.%s" % ext, dpi=300, width=6, height=4.8)


gCC = ggplot(data, aes('true', 'vanilla', color='tissue')) 
gCC += geom_abline(color='gray', linetype='dashed')
gCC += geom_point(size=2)
gCC += labs(color='Tissue', x='Actual age', y='Predicted age')
gCC += scale_color_manual(["#377EB8", "#E41A1C", "#A65628",
                           "#F781BF", "#4DAF4A", "#984EA3"])
gCC += theme_bw()
gCC += ggtitle("Methylation Age Prediction with Elastic Net")
gCC.save("figures/vanilla_true_comparison.%s" % ext, dpi=300, width=6, height=4.8)


# Figure 1D
# Results of pairwise experiment

pairwise_data_path = "output/pairwise/pairwise_accuracy.tsv"
pairwise = pd.read_csv(pairwise_data_path, delimiter="\t")

shuffled = pairwise.loc[pairwise['Shuffled'] == True, ['Source', 'Target', 'Elastic', 'Wenda']]
shuffled = shuffled.rename(columns={'Elastic':'Shuffled_Elastic', 'Wenda':'Shuffled_Wenda'})
signal = pairwise.loc[pairwise['Shuffled'] == False, ['Source', 'Target', 'Elastic', 'Wenda']]
signal = signal.rename(columns={'Elastic':'Signal_Elastic', 'Wenda':'Signal_Wenda'})

pairwise = pd.merge(signal, shuffled, on=['Source', 'Target'])
pairwise['Wenda_diff'] = pairwise['Signal_Wenda'] - pairwise['Shuffled_Wenda']
pairwise['Vanilla_diff'] = pairwise['Signal_Elastic'] - pairwise['Shuffled_Elastic']

gD = ggplot(pairwise, aes(x='Source', y='Target', fill='Wenda_diff')) + geom_tile()
gD += theme_classic() + theme(axis_text_x=element_text(angle=90))
gD += scale_fill_gradient2(low="#377EB8", high="#E41A1C", limits=[-0.8, 0.8])
gD += labs(fill='Accuracy (Signal - Shuffled)', x='Source Data', y='Target Data')
gD += ggtitle("TP53 Mutation Prediction with wenda_gpu")
gD.save("figures/pairwise_wenda.%s" % ext, dpi=300)

gDD = ggplot(pairwise, aes(x='Source', y='Target', fill='Vanilla_diff')) + geom_tile()
gDD += scale_fill_gradient2(low="#377EB8", high="#E41A1C", limits=[-0.8, 0.8])
gDD += theme_classic() + theme(axis_text_x=element_text(angle=90))
gDD += labs(fill='Accuracy (Signal - Shuffled)', x='Source Data', y='Target Data')
gDD += ggtitle("TP53 Mutation Prediction with Elastic Net")
gDD.save("figures/pairwise_vanilla.%s" % ext, dpi=300)


# Figure 1E
# Directly comparing accuracy for wenda and elastic net.
# This is just for the AACR poster

pairwise['Wenda_over_vanilla'] = pairwise['Signal_Wenda'] - pairwise['Signal_Elastic']
bettersame = pairwise.loc[pairwise['Wenda_over_vanilla'] >= 0, ]
print(bettersame.shape)

gE = ggplot(pairwise, aes(x='Signal_Elastic', y='Signal_Wenda'))
gE += geom_point(size=2)
gE += theme_bw()
gE += geom_abline()
gE += xlab("Elastic net accuracy") + ylab("wenda_gpu accuracy")
gE += ggtitle("TP53 Mutation Prediction Accuracy")
gE.save("figures/pairwise_scatterplot.%s" % ext, dpi=300)


# Combine all paper figures into one
def make_figure_panel(filename, scale, x_loc, y_loc):
    panel = sg.fromfile(filename)
    panel_size = (
            np.round(float(panel.root.attrib["width"][:-2]) * 1.33, 0),
            np.round(float(panel.root.attrib["height"][:-2]) * 1.33, 0)
            )

    print(f"original: {panel_size}")
    print(f"scaled:{(panel_size[0]*scale,panel_size[1]*scale)}")

    panel = panel.getroot()
    panel.moveto(x_loc, y_loc, scale)

    return panel


if ext == "svg":
    panel_1a = make_figure_panel(
            "figures/runtimes.svg",
            scale=0.85,
            x_loc=20,
            y_loc=20)

    panel_1b = make_figure_panel(
            "figures/software_correlation.svg",
            scale=0.85,
            x_loc=650,
            y_loc=20)

    panel_1c = make_figure_panel(
            "figures/vanilla_true_comparison.svg",
            scale=0.85,
            x_loc=20,
            y_loc=400)

    panel_1cc = make_figure_panel(
            "figures/wenda_true_comparison.svg",
            scale=0.85,
            x_loc=650,
            y_loc=400)

    panel_1d = make_figure_panel(
            "figures/pairwise_vanilla.svg",
            scale=0.85,
            x_loc=20,
            y_loc=800)

    panel_1dd = make_figure_panel(
            "figures/pairwise_wenda.svg",
            scale=0.85,
            x_loc=650,
            y_loc=800)

    panel_1a_label = sg.TextElement(20, 20, "A", size=16, weight="bold")
    panel_1b_label = sg.TextElement(650, 20, "B", size=16, weight="bold")
    panel_1c_label = sg.TextElement(20, 400, "C", size=16, weight="bold")
    panel_1d_label = sg.TextElement(20, 800, "D", size=16, weight="bold")

    figure_1 = sg.SVGFigure("1300", "1400")
    figure_1.append(
            [
                etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
                panel_1a,
                panel_1b,
                panel_1c,
                panel_1cc,
                panel_1d,
                panel_1dd,
                panel_1a_label,
                panel_1b_label,
                panel_1c_label,
                panel_1d_label
            ]
        )
    figure_1.save("figures/final.svg")
