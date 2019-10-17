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

# In[1]:


import math
import torch
import gpytorch
import sys
from matplotlib import pyplot as plt
sys.path.append('../')
from LBFGS import FullBatchLBFGS


# ----------------------------------------------------------------------------
# Begin Load Niklas Data.
# ----------------------------------------------------------------------------
niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)
n_data = inverseProblem.n_data

F = torch.as_tensor(inverseProblem.forward).detach()

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values[:, None]).detach()
d_obs_loc = torch.as_tensor(inverseProblem.data_points).detach()

cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()

# ----------------------------------------------------------------------------
# End Load Niklas Data.
# ----------------------------------------------------------------------------


import numpy as np

N = d_obs.shape[0]
train_x, train_y = d_obs_loc, d_obs

# normalize features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x = (train_x - mean) / std

# normalize labels
mean, std = train_y.mean(),train_y.std()
train_y = (train_y - mean) / std

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()

output_device = torch.device('cuda:0')

train_x, train_y = train_x.to(output_device), train_y.to(output_device)


# ## How many GPUs do you want to use?
# 
# In the next cell, specify the `n_devices` variable to be the number of GPUs you'd like to use. By default, we will use all devices available to us.

n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))


# ## GP Model + Training Code
# 
# In the next cell we define our GP model and training code. For this notebook, the only thing different from the Simple GP tutorials is the use of the `MultiDeviceKernel` to wrap the base covariance module. This allows for the use of multiple GPUs behind the scenes.


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices, F):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # Ambitious. We are working with lazy tensors, so may have to lazify F
        # and change the multiply operation.
        pushfwd_covar_x = torch.mm(F, torch.mm(covar_x, F.t()))
        pushfwd_mean_x = torch.mm(F, mean_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, pushfwd_covar_x)

def train(train_x,
          train_y,
          n_devices,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(train_x, train_y, likelihood, n_devices).to(output_device)
    model.train()
    likelihood.train()
    
    optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    
    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size),          gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)
            
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_training_iter, loss.item(),
                model.covar_module.module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            
            if fail:
                print('Convergence reached!')
                break
    
    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


# ## Automatically determining GPU Settings
# 
# In the next cell, we automatically determine a roughly reasonable partition or *checkpoint* size that will allow us to train without using more memory than the GPUs available have. Not that this is a coarse estimate of the largest possible checkpoint size, and may be off by as much as a factor of 2. A smarter search here could make up to a 2x performance improvement.

# In[6]:


import gc

def find_best_gpu_setting(train_x,
                          train_y,
                          n_devices,
                          output_device,
                          preconditioner_size
):
    N = train_x.size(0)
    
    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _ = train(train_x, train_y,
                         n_devices=n_devices, output_device=output_device,
                         checkpoint_size=checkpoint_size,
                         preconditioner_size=preconditioner_size, n_training_iter=1)
            
            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size

# Set a large enough preconditioner size to reduce the number of CG iterations run
preconditioner_size = 100
checkpoint_size = find_best_gpu_setting(train_x, train_y,
                                        n_devices=n_devices, 
                                        output_device=output_device,
                                        preconditioner_size=preconditioner_size)


# # Training

# In[7]:


model, likelihood = train(train_x, train_y,
                          n_devices=n_devices, output_device=output_device,
                          checkpoint_size=10000,
                          preconditioner_size=100,
                          n_training_iter=20)


# # Testing: Computing test time caches

# In[9]:


# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
    # Make predictions on a small number of test points to get the test time caches computed
    latent_pred = model(test_x[:2, :])
    del latent_pred  # We don't care about these predictions, we really just want the caches.


# # Testing: Computing predictions

# In[11]:


with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
    get_ipython().run_line_magic('time', 'latent_pred = model(test_x)')
    
test_rmse = torch.sqrt(torch.mean(torch.pow(latent_pred.mean - test_y, 2)))
print(f"Test RMSE: {test_rmse.item()}")


# In[10]:





# In[ ]:




