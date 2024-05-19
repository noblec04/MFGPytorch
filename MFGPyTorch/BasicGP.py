import math
import torch
import torch.nn as nn
import gpytorch
from matplotlib import pyplot as plt

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x1 = torch.linspace(0, 1, 5)
# True function is sin(2*pi*x) with Gaussian noise
train_y1 = ((6*train_x1 - 2)**2)*torch.sin(12*train_x1 - 4) + torch.randn(train_x1.size()) * math.sqrt(0.04)

train_x2 = torch.linspace(0, 1, 15)
# True function is sin(2*pi*x) with Gaussian noise
train_y2 = 0.5*((6*train_x2 - 2)**2)*torch.sin(12*train_x2 - 4) + 10*(train_x2) - 5 + torch.randn(train_x2.size()) * math.sqrt(0.04)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
model1 = ExactGPModel(train_x1, train_y1, likelihood1)

likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
model2 = ExactGPModel(train_x2, train_y2, likelihood2)

training_iter = 500


# Find optimal model hyperparameters
model1.train()
likelihood1.train()

# Use the adam optimizer
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

# Use the adam optimizer
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    # Output from model
    output1 = model1(train_x1)
    output2 = model2(train_x2)
    # Calc loss and backprop gradients
    loss1 = -mll1(output1, train_y1)
    loss1.backward()

    loss2 = -mll2(output2, train_y2)
    loss2.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss1.item(),
        model1.covar_module.base_kernel.lengthscale.item(),
        model1.likelihood.noise.item()
    ))

    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss2.item(),
        model2.covar_module.base_kernel.lengthscale.item(),
        model2.likelihood.noise.item()
    ))
    optimizer1.step()
    optimizer2.step()

# Get into evaluation (predictive posterior) mode
model1.eval()
likelihood1.eval()

model2.eval()
likelihood2.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred1 = likelihood1(model1(test_x))
    observed_pred2 = likelihood2(model2(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred1.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x1.numpy(), train_y1.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred1.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred2.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x2.numpy(), train_y2.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred2.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()