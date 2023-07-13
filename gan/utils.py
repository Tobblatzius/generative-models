import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import pylab as pl

# Helper for clamping all parameters in a network to (-limit, limit).
# Intended to be used with model.apply.
def clamp_params(module, clamp: float = 0.01):
    if hasattr(module, "weight"):
        w = module.weight.data
        w = w.clamp(-clamp, clamp)
        module.weight.data = w
    if hasattr(module, "bias"):
        b = module.bias.data
        b = b.clamp(-clamp, clamp)
        module.bias.data = b


# Helper for initializing the the parameters of a NN module.
# Intended to be used with model.apply.
# Parameter initialization
def init_params(module, init_weight_gain: float = 0.05, init_bias_const: float = 0.001):
    if hasattr(module, "weight"):
        torch.nn.init.xavier_uniform_(module.weight, gain=init_weight_gain)
    if hasattr(module, "bias"):
        torch.nn.init.constant_(module.bias, init_bias_const)


# For debugging, print all weights of the model
def print_params(model):
    all_params = np.array([])
    for param in model.parameters():
        this_param = torch.reshape(param.detach().cpu(), (-1,)).numpy()
        all_params = np.concatenate([all_params, this_param])
    print(all_params)


# Plot examples (generated or real) as a scatterplot. The input should be a
# Nx2 with N examples.
def plot_examples(examples):
    plt.clf()
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    plt.scatter(examples[:, 0], examples[:, 1], marker=".")
    plt.show()


def plot_inline(examples, generator_losses, critic_losses, step, means_x, means_y):
    pl.subplots(1, 3, figsize=(18, 6))
    pl.subplot(1, 3, 1)
    pl.plot(examples[:, 0], examples[:, 1], "b.", alpha=0.9)
    # plot the means of the dataset
    pl.plot(means_x, means_y, "r+")
    pl.xlim(-2, 2)
    pl.ylim(-2, 2)
    pl.title(f"Generated samples at step {step}")
    # plot the loss curves
    pl.subplot(1, 3, 2)
    pl.grid()
    pl.plot(generator_losses, label="Generator target")
    pl.legend()
    pl.subplot(1, 3, 3)
    pl.grid()
    pl.plot(critic_losses, label="Critic target")
    pl.legend()

    display.clear_output(wait=True)
    display.display(pl.gcf())
