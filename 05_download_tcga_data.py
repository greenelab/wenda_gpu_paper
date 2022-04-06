# Script to download TCGA expression and mutation data. This is pulled directly
# from https://github.com/greenelab/pancancer-evaluation/ under the directory
# 00_process_data/download_data.ipynb. Originally written by Jake Crawford.

import os
import pandas as pd
from urllib.request import urlretrieve

import pancancer_evaluation.config as cfg


url = 'http://api.gdc.cancer.gov/data/9a4679c3-855d-4055-8be9-3577ce10f66e'
name = 'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2-v2.geneExp.tsv'
exp_filepath = os.path.join(cfg.data_dir, name)
if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)

if not os.path.exists(exp_filepath):
    urlretrieve(url, exp_filepath)
else:
    print('Downloaded data file already exists, skipping download')

# Retrieve the downloaded expression data, update gene identifiers to entrez, and curate sample IDs. The script will also identify a balanced hold-out test set to compare projection performance into learned latent spaces across algorithms.

# ## Read TCGA Barcode Curation Information
# Extract information from TCGA barcodes - `cancer-type` and `sample-type`. See https://github.com/cognoma/cancer-data for more details

# Commit from https://github.com/cognoma/cancer-data/
sample_commit = 'da832c5edc1ca4d3f665b038d15b19fced724f4c'

url = 'https://raw.githubusercontent.com/cognoma/cancer-data/{}/mapping/tcga_cancertype_codes.csv'.format(sample_commit)
cancer_types_df = pd.read_csv(url,
                              dtype='str',
                              keep_default_na=False)

cancertype_codes_dict = dict(zip(cancer_types_df['TSS Code'],
                                 cancer_types_df.acronym))

url = 'https://raw.githubusercontent.com/cognoma/cancer-data/{}/mapping/tcga_sampletype_codes.csv'.format(sample_commit)
sample_types_df = pd.read_csv(url, dtype='str')

sampletype_codes_dict = dict(zip(sample_types_df.Code,
                                 sample_types_df.Definition))

# ## Read Entrez ID Curation Information
# 
# Load curated gene names from versioned resource. See https://github.com/cognoma/genes for more details

# Commit from https://github.com/cognoma/genes
genes_commit = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'

url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(genes_commit)
gene_df = pd.read_csv(url, sep='\t')

# Only consider protein-coding genes
gene_df = (
    gene_df.query("gene_type == 'protein-coding'")
)

# Load gene updater - define up to date Entrez gene identifiers where appropriate
url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(genes_commit)
updater_df = pd.read_csv(url, sep='\t')

old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id,
                             updater_df.new_entrez_gene_id))


# ## Read Gene Expression Data
tcga_expr_df = pd.read_csv(exp_filepath, index_col=0, sep='\t')


# ## Process gene expression matrix
# 
# This involves updating Entrez gene ids, sorting and subsetting.

# Set index as entrez_gene_id
tcga_expr_df.index = tcga_expr_df.index.map(lambda x: x.split('|')[1])

tcga_expr_df = (tcga_expr_df
    .dropna(axis='rows')
    .rename(index=old_to_new_entrez)
    .groupby(level=0).mean()
    .transpose()
    .sort_index(axis='rows')
    .sort_index(axis='columns')
)

tcga_expr_df.index.rename('sample_id', inplace=True)

# Update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
tcga_expr_df.index = tcga_expr_df.index.str.slice(start=0, stop=15)
tcga_expr_df = tcga_expr_df.loc[~tcga_expr_df.index.duplicated(), :]

# Filter for valid Entrez gene identifiers
tcga_expr_df = tcga_expr_df.loc[:, tcga_expr_df.columns.isin(gene_df.entrez_gene_id.astype(str))]


expr_file = os.path.join(cfg.data_dir, 'tcga_expression_matrix_processed.tsv.gz')
tcga_expr_df.to_csv(expr_file, sep='\t', compression='gzip', float_format='%.3g')


# ## Process TCGA cancer-type and sample-type info from barcodes
# 
# Cancer-type includes `OV`, `BRCA`, `LUSC`, `LUAD`, etc. while sample-type includes `Primary`, `Metastatic`, `Solid Tissue Normal`, etc.
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.
# 
# The goal is to use this info to stratify training (90%) and testing (10%) balanced by cancer-type and sample-type. 

# Extract sample type in the order of the gene expression matrix
tcga_id = pd.DataFrame(tcga_expr_df.index)

# Extract the last two digits of the barcode and recode sample-type
tcga_id = tcga_id.assign(sample_type = tcga_id.sample_id.str[-2:])
tcga_id.sample_type = tcga_id.sample_type.replace(sampletype_codes_dict)

# Extract the first two ID numbers after `TCGA-` and recode cancer-type
tcga_id = tcga_id.assign(cancer_type = tcga_id.sample_id.str[5:7])
tcga_id.cancer_type = tcga_id.cancer_type.replace(cancertype_codes_dict)

# Append cancer-type with sample-type to generate stratification variable
tcga_id = tcga_id.assign(id_for_stratification = tcga_id.cancer_type.str.cat(tcga_id.sample_type))

# Get stratification counts - function cannot work with singleton strats
stratify_counts = tcga_id.id_for_stratification.value_counts().to_dict()

# Recode stratification variables if they are singletons
tcga_id = tcga_id.assign(stratify_samples_count = tcga_id.id_for_stratification)
tcga_id.stratify_samples_count = tcga_id.stratify_samples_count.replace(stratify_counts)
tcga_id.loc[tcga_id.stratify_samples_count == 1, "stratify_samples"] = "other"

# Write out files for downstream use
file = os.path.join(cfg.data_dir, 'tcga_sample_identifiers.tsv')

(
    tcga_id.drop(['stratify_samples', 'stratify_samples_count'], axis='columns')
    .to_csv(file, sep='\t', index=False)
)

cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.data_dir, 'tcga_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)


# ## Subsample expression dataframe for unit testing

# We want the subsampled data to have representation of most cancer types
# We can use stratified cross-validation to make sure of this
from sklearn.model_selection import train_test_split
_, subsample_df = train_test_split(tcga_expr_df,
                                   test_size=0.1,
                                   random_state=cfg.default_seed,
                                   stratify=tcga_id.stratify_samples_count)

# Also subsample genes, otherwise file size blows up
subsample_df = subsample_df.sample(n=100, axis=1, random_state=cfg.default_seed)

if not os.path.exists(cfg.test_data_dir):
    os.makedirs(cfg.test_data_dir)
subsample_df.to_csv(cfg.test_expression, sep='\t', 
                    compression='gzip', float_format='%.3g')
