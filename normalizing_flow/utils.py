## Standard libraries
import math
import os
import os.path as osp
from typing import Iterable

## Imports for plotting
import matplotlib.pyplot as plt
import numpy as np

# PyTorch Lightning
import pytorch_lightning as pl

## PyTorch Data Loading
import torch
import torch.utils.data as data
import torchvision
from IPython.display import Image
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.datasets import MNIST

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "data"


# Transformations applied on each image => bring them into a numpy array
# Note that we keep them in the range 0-255 (integers)
def image_to_numpy(img):
    img = np.array(img, dtype=np.uint8)  # do we need to use int32 here?
    img = img[..., None]  # Make image [28, 28, 1]
    return img


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], torch.Tensor):
        return (torch.stack(batch)).numpy()
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_data(dataset_name):
    # Convert images from 0-1 to 0-255 (integers)
    def discretize(sample):
        return (sample * 255).to(torch.int32)

    # Transformations applied on each image => make them a tensor and discretize
    transform = transforms.Compose([transforms.ToTensor(), discretize])

    if dataset_name == "mnist":
        train_dataset = MNIST(
            root=DATASET_PATH, train=True, transform=transform, download=True
        )
        pl.seed_everything(42)
        train_set, val_set = data.random_split(train_dataset, [50000, 10000])
        test_set = MNIST(
            root=DATASET_PATH, train=False, transform=transform, download=True
        )
    elif dataset_name == "celeba":
        celeba_dict = np.load("celeba_28x28_dict.npy", allow_pickle=True)[()]
        # Train
        img_train = torch.tensor(celeba_dict["celeba_train_img"].transpose(0, 3, 1, 2))
        attr_train = torch.tensor(celeba_dict["celeba_train_attr"])
        train_set = data.TensorDataset(img_train, attr_train)
        # Val
        img_val = torch.tensor(celeba_dict["celeba_val_img"].transpose(0, 3, 1, 2))
        attr_val = torch.tensor(celeba_dict["celeba_val_attr"])
        val_set = data.TensorDataset(img_val, attr_val)
        # Test
        img_test = torch.tensor(celeba_dict["celeba_test_img"].transpose(0, 3, 1, 2))
        attr_test = torch.tensor(celeba_dict["celeba_test_attr"])
        test_set = data.TensorDataset(img_test, attr_test)
    else:
        raise NotImplementedError

    train_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=False, drop_last=False
    )
    val_loader = data.DataLoader(
        val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4
    )
    test_loader = data.DataLoader(
        test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4
    )
    return (train_set, train_loader, val_loader, test_loader)


def show_imgs(imgs, title=None, row_size=20):
    # Form a grid of pictures (we use max. 8 columns)
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    is_int = (
        imgs.dtype == torch.int32
        if isinstance(imgs, torch.Tensor)
        else imgs[0].dtype == torch.int32
    )
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs / nrow))
    imgs = torchvision.utils.make_grid(
        imgs, nrow=nrow, pad_value=128 if is_int else 0.5
    )
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation="nearest")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


# Decorator that creates a directory if it doesn't exist, based on "output_dir" argument
def create_output_dir(func):
    def wrapper(*args, **kwargs):
        output_dir = kwargs.get("output_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return func(*args, **kwargs)

    return wrapper


@create_output_dir
def vis_1d(
    data,
    flow=None,
    epoch: int = 0,
    output_dir: str = None,
    xlim=(-8, 16),
    ylim=(0, 0.35),
):
    plt.hist(data.numpy(), bins=30, alpha=0.5, density=True, label="True")
    if flow is not None:
        samples = flow.sample((1000,)).detach().numpy()
        plt.hist(samples, bins=30, alpha=0.5, density=True, label="Sampled")
    plt.legend(loc="upper right")
    plt.title(f"Epoch {epoch}")
    plt.xlim(xlim)
    plt.ylim(ylim)
    if output_dir:
        out_path = osp.join(output_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


@create_output_dir
def vis_2d(
    data,
    flow=None,
    epoch: int = 0,
    output_dir: str = None,
    xlim=(-4, 4),
    ylim=(-4, 4),
):
    # Visualize the learned distribution
    n_points = 1000
    grid = torch.linspace(-4, 4, n_points)
    xx, yy = torch.meshgrid(grid, grid)
    zz = torch.stack([xx, yy], dim=-1).view(-1, 2)

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Epoch {epoch}")
    plt.subplot(131)
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    plt.title("True Distribution")
    plt.xlim(xlim)
    plt.ylim(ylim)

    if flow is not None:
        with torch.no_grad():
            zz_log_prob = flow.log_prob(zz).exp().view(n_points, n_points)

        plt.subplot(132)
        plt.contourf(xx.numpy(), yy.numpy(), zz_log_prob.numpy())
        plt.title("Learned Distribution")
        plt.xlim(xlim)
        plt.ylim(ylim)

        # Visualize samples from the learned distribution
        with torch.no_grad():
            plt.subplot(133)
            samples = flow.sample((10000, 2))
            plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
            plt.title("Generated Samples")
            plt.xlim(xlim)
            plt.ylim(ylim)

    if output_dir:
        plt.savefig(osp.join(output_dir, f"epoch_{epoch:04d}.png"))
    else:
        plt.show()
    plt.close()


def create_gif(output_dir, epochs: Iterable[int]):
    images = [PILImage.open(osp.join(output_dir, f"epoch_{i:04d}.png")) for i in epochs]
    gif_path = osp.join(output_dir, "training_evolution.gif")
    images[0].save(
        gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
    )
    return Image(filename=gif_path)


def print_num_params(model):
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Number of parameters: {:,}".format(num_params))
