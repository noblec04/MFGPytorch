from kan import *
import torch

# initialize KAN with G=3
model = KAN(width=[3,8,1], grid=5, k=3)

# create dataset
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + (x[:,[1]] - x[:,[2]])**2 + x[:,[2]])
dataset = create_dataset(f, n_var=3)

model.train(dataset, opt="LBFGS", steps=50);

model.plot()
plt.show()

# initialize a more fine-grained KAN with G=10
#model2 = KAN(width=[3,4,1], grid=10, k=3)
# initialize model2 from model
#model2.initialize_from_another_model(model, dataset['train_input']);

#model2.train(dataset, opt="LBFGS", steps=20);

#model2.plot()
#plt.show()

#grids = np.array([100])

#train_losses = []
#test_losses = []
#steps = 50
#k = 3

#for i in range(grids.shape[0]):
#   if i == 0:
#        model = KAN(width=[3,4,1], grid=grids[i], k=k)
#    if i != 0:
#        model = KAN(width=[3,4,1], grid=grids[i], k=k).initialize_from_another_model(model, dataset['train_input'])
#    results = model.train(dataset, opt="LBFGS", steps=steps, stop_grid_update_step=30)
#    train_losses += results['train_loss']
#    test_losses += results['test_loss']

#plt.figure()
#plt.plot(train_losses)
#plt.plot(test_losses)
#plt.legend(['train', 'test'])
#plt.ylabel('RMSE')
#plt.xlabel('step')
#plt.yscale('log')

#n_params = 3 * grids
#train_vs_G = train_losses[(steps-1)::steps]
#test_vs_G = test_losses[(steps-1)::steps]

#plt.plot(n_params, train_vs_G, marker="o")
#plt.plot(n_params, test_vs_G, marker="o")
#plt.plot(n_params, 100*n_params**(-4.), ls="--", color="black")
#plt.xscale('log')
#plt.yscale('log')
#plt.legend(['train', 'test', r'$N^{-4}$'])
#plt.xlabel('number of params')
#plt.ylabel('RMSE')

#plt.show()