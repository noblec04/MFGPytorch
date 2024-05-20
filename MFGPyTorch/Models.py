import torch
import torch.nn as nn
import gpytorch
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x1 = torch.linspace(0, 1, 5)
# True function is sin(2*pi*x) with Gaussian noise
train_y1 = ((6*train_x1 - 2)**2)*torch.sin(12*train_x1 - 4)

noises_1 = torch.ones(4) * 0.001

train_x2 = torch.linspace(0, 1, 20)
# True function is sin(2*pi*x) with Gaussian noise
train_y2 = 0.5*((6*train_x2 - 2)**2)*torch.sin(12*train_x2 - 4) + 10*(train_x2) - 5 + torch.randn(train_x2.size()) * torch.sqrt(torch.tensor(0.1))

# We will use the simplest form of GP model, exact inference
class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=gpytorch.means.ZeroMean(),kernel=gpytorch.kernels.MaternKernel(nu=2.5)):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def condition(self,trainx,trainy,training_iter = 500,lr=0.1):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.forward(trainx)
            # Calc loss and backprop gradients
            loss = -mll(output, trainy)
            loss.backward()

            print('Iter %d/%d - Loss: %.3f ' % ( # lengthscale: %.3f  noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                #self.covar_module.base_kernel.lengthscale.item(),
                #self.likelihood.noise.item()
            ))
            optimizer.step()

    def freeze(self):
        
        for param in self.parameters():
            param.requires_grad = False

        for param in self.likelihood.parameters():
            param.requires_grad = False

    def unfreeze(self):
        
        for param in self.parameters():
            param.requires_grad = True

        for param in self.likelihood.parameters():
            param.requires_grad = True
    

def maxvar(model):
        
        xx = torch.tensor([0.5],dtype=torch.float32,requires_grad=True)

        optimizer = torch.optim.Adam([xx], lr=0.2)

        for i in range(100):
            
            optimizer.zero_grad()

            v = -1*model(xx)[1]

            v.backward()

            print('Iter %d/%d - loc: %.3f - Loss: %.3f ' % ( # lengthscale: %.3f  noise: %.3f' % (
                i + 1, 100, xx.item(), -1*v.item(),
                #self.covar_module.base_kernel.lengthscale.item(),
                #self.likelihood.noise.item()
            ))

            optimizer.step()
        
        return xx, -1*v


class MFGP(nn.Module):
    def __init__(self, x1, y1, x2, y2):
        
        super().__init__()

        likelihood1 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_1, learn_additional_noise=False)
        likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
        
        self.GP2 = GP(x2, y2, likelihood2)

        self.GP2.train()
        self.GP2.likelihood.train()

        self.GP2.condition(x2,y2)

        self.GP2.eval()
        self.GP2.likelihood.eval()

        self.GP2.freeze()

        pred2 = self.GP2(x1).mean.detach().numpy()

        self.rho = torch.tensor(LinearRegression().fit(pred2.reshape((-1, 1)),train_y1.detach().numpy()).coef_)

        self.GPd = GP(x1, y1 - self.rho*pred2, likelihood1)

        self.GPd.train()
        self.GPd.likelihood.train()

        self.GPd.condition(x1, y1 - self.rho*pred2,lr=0.2)

        self.GPd.eval()
        self.GPd.likelihood.eval()

        self.GPd.freeze()

    def forward(self,x):
        with gpytorch.settings.fast_pred_var():
            test_noises = torch.ones_like(x) * 0.001
            observed_pred2 = self.GP2.likelihood(self.GP2(x)) #, noise=test_noises
            observed_predd = self.GPd.likelihood(self.GPd(x), noise=test_noises)

        mean_pred = self.rho*observed_pred2.mean + observed_predd.mean
        var_pred = (self.rho**2)*observed_pred2.variance + observed_predd.variance
       
        return mean_pred, var_pred

MFmodel = MFGP(train_x1,train_y1,train_x2,train_y2)

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad():
    test_x = torch.linspace(0, 1, 51)
    test_noises = torch.ones(51) * 0.001
    observed_pred2 = MFmodel.GP2.likelihood(MFmodel.GP2(test_x))
    mean_mf, var_mf = MFmodel(test_x)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower2, upper2 = observed_pred2.confidence_region()

    # Plot training data as black stars
    ax.plot(train_x1.numpy(), train_y1.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), mean_mf.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), mean_mf.numpy() - 2*torch.sqrt(var_mf).numpy(), mean_mf.numpy() + 2*torch.sqrt(var_mf).numpy(), alpha=0.5)

    ax.plot(train_x2.numpy(), train_y2.numpy(), 'y*')
    ax.plot(test_x.numpy(), observed_pred2.mean.numpy(), 'g')
    ax.fill_between(test_x.numpy(), lower2.numpy(), upper2.numpy(), alpha=0.5)

    ax.legend(['Observed HF Data', 'MF Mean', 'MF Confidence', 'Observed LF Data', 'LF Mean', 'LF Confidence'])
    plt.show()

x,v = maxvar(MFmodel)

print(x)
print(v)