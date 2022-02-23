import gpytorch
import math
import torch
#import pyro
#import pyro.distributions as dist
#from pyro.nn import PyroModule, PyroSample

from LBFGS import FullBatchLBFGS
from scipy.stats import norm


'''class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[torch.nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
'''

class SpectralDeltaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralDeltaGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.SpectralDeltaKernel(
                num_dims=train_x.size(-1),
                num_deltas=num_deltas,
        )
        base_covar_module.intialize_from_data(train_x[0], train_y[0])
        self.covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_size):
        super(BatchGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.LinearKernel(batch_shape=torch.Size([batch_size]))

        # Initialize variance to 1/d so that inner products between data points are ~ 1.
        # Unscaled inner products in train_x are so large that we lose precision.
        self.covar_module.variance = 1. / train_x.size(-1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()
        
        # Initialize variance to 1/d so that inner products between data points are ~ 1.
        # Unscaled inner products in train_x are so large that we lose precision.
        self.covar_module.variance = 1. / train_x.size(-1)
  
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model_sdk(model, likelihood, x, y, learning_rate,
                training_iter=10):
    lbfgs = FullBatchLBFGS(model.parameters(), lr=learning_rate)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    

def train_model_bfgs_batch(model, likelihood, x, y, learning_rate,
                training_iter=10):
    lbfgs = FullBatchLBFGS(model.parameters(), lr=learning_rate)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        model.zero_grad()
        output = model(x)
        loss = -mll(output, y).sum()
        return loss

    loss = closure()
    loss.backward()

    for i in range(training_iter):
        options = {"closure": closure, "current_loss": loss, "max_ls": 10}
        loss, _, lr, _, F_eval, G_eval, _, fail = lbfgs.step(options)

        if fail:
            break

    return model, likelihood


def train_model_bfgs(model, likelihood, x, y, learning_rate,
                training_iter=10):
    lbfgs = FullBatchLBFGS(model.parameters(), lr=learning_rate)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        model.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        return loss

    loss = closure()
    loss.backward()

    for i in range(training_iter):
        options = {"closure": closure, "current_loss": loss, "max_ls": 10}
        loss, _, lr, _, F_eval, G_eval, _, fail = lbfgs.step(options)

        if fail:
            break

    return model, likelihood


def getConfidence(model, likelihood, x, y):
    with gpytorch.settings.fast_pred_var():
        f_preds = likelihood(model(x))
    mu = f_preds.mean
    sigma_sq = f_preds.variance
    sigma_sq = torch.sqrt(sigma_sq)
    res_normed = (y - mu) / sigma_sq
    res_normed = res_normed.cpu().detach().numpy()
    confidences = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))
    mu = mu.cpu().detach().numpy()
    sigma_sq = sigma_sq.cpu().detach().numpy()
    return mu, sigma_sq ** 2, confidences


