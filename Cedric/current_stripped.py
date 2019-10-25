#!/usr/bin/env python
# coding: utf-8

# # Exact GP Regression with Multiple GPUs and Kernel Partitioning
# 
# In this notebook, we'll demonstrate training exact GPs on large datasets using two key features from the paper https://arxiv.org/abs/1903.08114: 
# 
# 1. The ability to distribute the kernel matrix across multiple GPUs, for additional parallelism.
# 2. Partitioning the kernel into chunks computed on-the-fly when performing each MVM to reduce memory usage.
# 
# We'll be using the `protein` dataset, which has about 37000 training examples. The techniques in this notebook can be applied to much larger datasets, but the training time required will depend on the computational resources you have available: both the number of GPUs available and the amount of memory they have (which determines the partition size) have a significant effect on training time.


import math
import torch
import torch.optim
import gpytorch
import sys
sys.path.append('../')


# ----------------------------------------------------------------------------
# Begin Load Niklas Data.
# ----------------------------------------------------------------------------
from volcapy.inverse.inverse_problem import InverseProblem
# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
niklas_data_path = "/home/ubuntu/Dev/Data/Cedric.mat"
# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)
n_data = inverseProblem.n_data


# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values).float()
d_obs_loc = torch.as_tensor(inverseProblem.data_points).float()
cells_coords = torch.as_tensor(inverseProblem.cells_coords).float()

# ----------------------------------------------------------------------------
# End Load Niklas Data.
# ----------------------------------------------------------------------------


# ## Normalization and train/test Splits
# 
# In the next cell, we split the data 80/20 as train and test, and do some basic z-score feature normalization.
import numpy as np

train_x = cells_coords
train_y = d_obs
n_train = train_x.shape[0]
print(f"Training on {n_train} datapoints.")

"""
# normalize features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x = (train_x - mean) / std

# normalize labels
mean, std = train_y.mean(),train_y.std()
train_y = (train_y - mean) / std
"""

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()

output_device = torch.device('cuda:0')

train_x, train_y = train_x.to(output_device), train_y.to(output_device)

F = torch.as_tensor(inverseProblem.forward).float()
F = F.to(output_device)

n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))


# ## GP Model + Training Code
# 
# In the next cell we define our GP model and training code. For this notebook, the only thing different from the Simple GP tutorials is the use of the `MultiDeviceKernel` to wrap the base covariance module. This allows for the use of multiple GPUs behind the scenes.


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices, F):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel())
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

        self.F = F
    
    def forward(self, x):
        mean_x = torch.mm(self.F, self.mean_module(x)[:, None]).squeeze()
        covar_x = torch.mm(F, self.covar_module(x)._matmul(F.t()))

        # Add noise.
        covar_x = covar_x + 0.1**2 * torch.eye(covar_x.shape[0]).to(output_device)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(train_x,
          train_y,
          n_devices,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(train_x, train_y, likelihood, n_devices, F).to(output_device)
    model.train()
    likelihood.train()

    hypers = {
            'likelihood.noise_covar.noise': torch.tensor(0.1).to(output_device).float(),
            'covar_module.module.base_kernel.lengthscale': torch.tensor(300.0).to(output_device).float(),
            'covar_module.module.outputscale': torch.tensor(200.0**2).to(output_device).float(),
            }
    model.initialize(**hypers)
    print(
            model.likelihood.noise_covar.noise.item(),
            model.covar_module.module.base_kernel.lengthscale.item(),
            model.covar_module.module.outputscale.item()
            )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=200.0)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), gpytorch.settings.max_preconditioner_size(preconditioner_size):
        # Warm up.
        print("Forward pass.")
        output = model(train_x)
        print("Forward pass done.")
    
    with (gpytorch.beta_features.checkpoint_kernel(checkpoint_size),
            gpytorch.settings.max_preconditioner_size(preconditioner_size)):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            loss = optimizer.step(closure=closure)
            
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f standard deviation: %.3f   noise: %.3f' % (
                i + 1, n_training_iter, loss.item(),
                model.covar_module.module.base_kernel.lengthscale.item(),
                np.sqrt(model.covar_module.module.outputscale.item()),
                model.likelihood.noise.item()
            ))
    
    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


# FIRST CALL #
# Set a large enough preconditioner size to reduce the number of CG iterations run
preconditioner_size = 1000
print("Calling first call.")

model, likelihood = train(train_x, train_y,
        n_devices=n_devices, output_device=output_device,
        checkpoint_size=30000,
        preconditioner_size=preconditioner_size, n_training_iter=20)


# # Testing: Computing test time caches
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
