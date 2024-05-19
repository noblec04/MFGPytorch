import math
import torch
import torch.nn as nn
import gpytorch
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x1 = torch.linspace(0, 1, 4)
# True function is sin(2*pi*x) with Gaussian noise
train_y1 = ((6*train_x1 - 2)**2)*torch.sin(12*train_x1 - 4)

noises_1 = torch.ones(4) * 0.001

train_x2 = torch.linspace(0, 1, 20)
# True function is sin(2*pi*x) with Gaussian noise
train_y2 = 0.5*((6*train_x2 - 2)**2)*torch.sin(12*train_x2 - 4) + 10*(train_x2) - 5 + torch.randn(train_x2.size()) * math.sqrt(0.04)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(1)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood1 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_1, learn_additional_noise=False)#GaussianLikelihood()
#likelihood2 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_2, learn_additional_noise=False)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()

GP2 = ExactGPModel(train_x2, train_y2, likelihood2)

training_iter = 500

# Find optimal model hyperparameters
GP2.train()
likelihood2.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(GP2.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, GP2)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = GP2(train_x2)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y2)
    loss.backward()

    print('Iter %d/%d - Loss: %.3f ' % (#  lengthscale: %.3f  noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        #GP2.covar_module.base_kernel.lengthscale.item(),
        #GP2.likelihood.noise.item()
    ))
    optimizer.step()

GP2.eval()
likelihood2.eval()

pred2 = GP2(train_x1).mean.detach().numpy()

rho = torch.tensor(LinearRegression().fit(pred2.reshape((-1, 1)),train_y1.detach().numpy()).coef_)

GPd = ExactGPModel(train_x1, train_y1 - rho*pred2, likelihood1)

training_iter = 500


# Find optimal model hyperparameters
GPd.train()
likelihood1.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(GPd.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, GPd)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    outputd = GPd(train_x1)
    output2 = GP2(train_x1)
    outputm = rho*output2.mean + outputd.mean
    outputcov = (rho**2)*output2.covariance_matrix + outputd.covariance_matrix

    output = gpytorch.distributions.MultivariateNormal(outputm, outputcov)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y1)
    loss.backward()

    print('Iter %d/%d - Loss: %.3f' % (  #lengthscale: %.3f' % (
        i + 1, training_iter, loss.item(),
        #GPd.covar_module.base_kernel.lengthscale.item(),
    ))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
GPd.eval()
likelihood1.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad():#, gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    test_noises = torch.ones(51) * 0.001
    observed_pred2 = likelihood2(GP2(test_x)) #, noise=test_noises
    observed_predd = likelihood1(GPd(test_x), noise=test_noises)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower2, upper2 = observed_pred2.confidence_region()
    lowerd, upperd = observed_predd.confidence_region()

    lower = rho*lower2 + lowerd
    upper = rho*upper2 + upperd

    # Plot training data as black stars
    ax.plot(train_x1.numpy(), train_y1.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), rho*observed_pred2.mean.numpy() + observed_predd.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()