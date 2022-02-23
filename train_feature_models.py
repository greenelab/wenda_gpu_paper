import os
import argparse
import gpytorch
import numpy as np
import pandas as pd
import torch
import model_train_helper

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prefix', default="handl", help="Dataset identifier, will be used to name folders for intermediate and output data.")
parser.add_argument('-s', '--start', type=int, default=0, help="What feature number to start training on. This is needed for batch training.")
parser.add_argument('-r', '--range', type=int, default=100, help="How many feature models to train. This is needed for batch training.")
args = parser.parse_args()

# Set torch to run with float32 instead of float64, which exponentially
# increases speed with neglibile decrease in precision.
dtype = torch.float32
device = 'cuda'

# Load data
source_file = os.path.join("data", args.prefix, "source_data.tsv")
source_table = pd.read_csv(source_file, sep="\t", header=None)
source_matrix = np.asfortranarray(source_table.values)

target_file = os.path.join("data", args.prefix, "target_data.tsv")
target_table = pd.read_csv(target_file, sep="\t", header=None)
target_matrix = np.asfortranarray(target_table.values)

# Normalize based on source data, with some noise to avoid dividing by 0
epsilon = 1e-6
means = np.mean(source_matrix, axis=0)
stds = np.std(source_matrix, axis=0) + epsilon

normed = (source_matrix - means) / stds
normed_source_matrix = normed

normed = (target_matrix - means) / stds
normed_target_matrix = normed

# Since we're doing side-by-side runs of wenda_gpu and wenda_orig,
# we can pull from the same data directory but need to store output
# (feature models and confidence scores) in separate folders.
output_prefix = args.prefix + "_wenda_gpu"

# Make directory to store feature models
output_dir = os.path.join("feature_models", output_prefix)
os.makedirs(output_dir, exist_ok=True)

# Make directory to store confidence scores from target data
conf_dir = os.path.join("confidences", output_prefix)
os.makedirs(conf_dir, exist_ok=True)

first_model = args.start
total_features = source_matrix.shape[1]

for i in range(args.range):
    #Prevent out of range error if start + range > total number of features
    feature_number = first_model + i
    if feature_number >= total_features:
        break
    try:

        # If confidences have already been calculated, skip feature
        conf_file = os.path.join(conf_dir, "model_%s_confidence.txt" % feature_number)
        if os.path.isfile(conf_file):
            continue

        # Split out feature to predict using all other features
        train_y = torch.from_numpy(normed_source_matrix[:, feature_number])
        train_y = train_y.to(device=device, dtype=dtype)
        train_x = torch.from_numpy(np.delete(normed_source_matrix, feature_number, 1)).squeeze(-1)
        train_x = train_x.to(device=device, dtype=dtype)
        test_y = torch.from_numpy(normed_target_matrix[:, feature_number])
        test_y = test_y.to(device=device, dtype=dtype)
        test_x = torch.from_numpy(np.delete(normed_target_matrix, feature_number, 1)).squeeze(-1)
        test_x = test_x.to(device=device, dtype=dtype)

        # Train model if it has not been previously generated
        modelfile = os.path.join(output_dir, "model_%s.pth" % feature_number)
        if os.path.isfile(modelfile) is False:
        
            # Initialize model and likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
            model = model_train_helper.ExactGPModel(train_x, train_y, likelihood).to(device=device, dtype=dtype)

            model.train()
            likelihood.train()

            with gpytorch.settings.max_cholesky_size(100000), gpytorch.settings.cholesky_jitter(1e-5):
                model, likelihood = model_train_helper.train_model_bfgs(
                    model, likelihood, train_x, train_y, learning_rate=1., training_iter=15
                    )

            model.eval()
            likelihood.eval()

            # Save feature model
            torch.save(model.state_dict(),os.path.join(output_dir, "model_%s.pth" % feature_number))
    
        # If model was previously generated, load for confidence score calculation
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
            state_dict = torch.load(modelfile)
            model = model_train_helper.ExactGPModel(train_x, train_y, likelihood).to(device=device, dtype=dtype)
            model.load_state_dict(state_dict)

            model.eval()
            likelihood.eval()

        # Write out confidence scores
        mean, var, conf = model_train_helper.getConfidence(model, likelihood, test_x, test_y)
        np.savetxt(conf_file, conf, fmt='%.10f')

    # If the model is unable to be trained, we consider it safe to assign that
    # feature a confidence score of 0 for all the target data. This will
    # increase the penalty on the untrainable model's feature in the elastic net.
    except gpytorch.utils.errors.NotPSDError:
        conf = np.zeros(test_x.shape[0])
        np.savetxt(conf_file, conf, fmt='%i')
