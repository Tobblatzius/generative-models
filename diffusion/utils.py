import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from collections.abc import Callable
from IPython.display import Image
from PIL import Image as PILImage
from torchvision import transforms
import torch.nn as nn
import scipy.stats as st
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from livelossplot import PlotLosses


# Function to plot the 2D distribution
def plot_2D(
    data,
    xlim=(-4, 4),
    ylim=(-4, 4)):

    # Visualize the 2D distribution
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=1)
    plt.title("True Distribution")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()
    plt.axis("equal")

# Function to get the 2D densities
def get_2D_densities(predicted_points):
    
    xmin = -4
    xmax = 4
    ymin = -4
    ymax = 4

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    values = predicted_points.numpy()
    values = np.vstack([values[:,0], values[:,1]])
    kernel = st.gaussian_kde(values)
    
    f = np.reshape(kernel(positions).T, xx.shape)

    return f, xx, yy


# Function to visualize the steps of forward/backward diffusion on 2D points
def visualize_2D_steps(x_ts: list,
                       timesteps: list[int],
                       output_dir: Path,
                       plot_every = 1,
                       plot_densities: bool=True):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    num_plots = 1
    if plot_densities:
        num_plots += 1

    print("Creating plots for visualizations... Computing densities is slow")
    for t, x_t in tqdm(zip(timesteps, x_ts)):

        if x_t.device != torch.device("cpu"):
            x_t = x_t.cpu()

        if t % plot_every != 0:
            continue

        fig, ax = plt.subplots(1, num_plots, figsize=(10, 5))

        # Plot the points and save the figure
        ax[0].scatter(x_t[:, 0], x_t[:, 1], s=0.5)
        ax[0].set_title(f"Points at timestep = {t}")
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_xlim(-4, 4)
        ax[0].set_ylim(-4, 4)   

        # Plot the density (slow)
        if plot_densities:

            f, xx, yy = get_2D_densities(x_t)
            cfset = ax[1].contourf(xx, yy, f, cmap='coolwarm')

            ax[1].imshow(np.rot90(f), cmap='coolwarm', extent=[-4, 4, -4, 4])
            ax[1].set_title(f"Emperical prob. density at timstep = {t}")
            cset = ax[1].contour(xx, yy, f, colors='k')
            ax[1].clabel(cset, inline=1, fontsize=10)
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')
            ax[1].set_xlim(-4, 4)
            ax[1].set_ylim(-4, 4)

        plt.savefig(os.path.join(output_dir, f"timestep_{t}.png"))
        plt.close()

# Function to visualize the steps of forward/backward diffusion on Images
def visualize_sampled_images(samples: list,
                          output_dir : Path):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(samples[-1][i * 10 + j].squeeze(), cmap="gray")
            axs[i, j].axis("off")

    plt.savefig(os.path.join(output_dir, "fashion_generated_samples.png"))
    plt.show()


def get_concat_h(ims):

    dst = Image.new('RGB', (sum(im.width for im in ims), ims[0].height ))
    
    w = 0
    for i,im in enumerate(ims):
        
        dst.paste(im, (w, 0))
        w+=im.width
    
    return dst


# Function to visualize the steps of forward/backward diffusion on Images
def save_image_steps(xs_ts: list[list],
                     timesteps: list[int],
                            output_dir : Path):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    num_images = len(xs_ts)
    num_steps = None

    for x_ts in xs_ts:
        if num_steps is not None:
            assert len(x_ts) == num_steps
        else:
            num_steps = len(x_ts)

    for i,t in enumerate(timesteps):

        ims = []

        for x_ts in enumerate(xs_ts):
            
            pil_image_single = PILImage.fromarray(x_ts[i].squeeze().numpy(), mode="F")
            ims.append(pil_image_single)

        pil_image_concatenated = get_concat_h(ims)

    pil_image_concatenated.save(os.path.join(output_dir, f"generated_timestep_{t}.png"))

# Function to create a gif from the saved images
def create_gif(output_dir,
               prefix:str,
               reverse: bool = False,
               gif_name: str = "animation.gif"):
    
    imnames_all = os.listdir(output_dir)
    ts = []
    for imname in imnames_all:
        if not imname.startswith(prefix):
            continue
        t = int(imname.split("_")[-1].split(".")[0])
        ts.append(t)
    
    ts = np.array(ts)
    ts.sort()

    images = [PILImage.open(osp.join(output_dir, f"{prefix}_{t}.png")) for t in ts]
    if reverse == True:
        images.reverse()

    gif_path = osp.join(output_dir, gif_name)

    images[0].save(
        gif_path, save_all=True, append_images=images[1:], duration=150, loop=0
    )
    
    return Image(filename=gif_path)

# Function to visualize the steps of forward/backward diffusion on Images
def get_fashion_data(batch_size: int):
    dataset = load_dataset("fashion_mnist")

    # define image transformations
    def transforms_f(train=True):
        transform_list = [transforms.RandomHorizontalFlip()] if train else []
        transform = Compose(transform_list +
                            [
                                transforms.ToTensor(),
                                transforms.Lambda(lambda t: (t * 2) - 1)
                            ])

        def f(examples):
            examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
            del examples["image"]

            return examples

        return f

    # transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
    transformed_dataset_train = dataset.with_transform(transforms_f(train=True))
    transformed_dataset_val = dataset.with_transform(transforms_f(train=False))

    # create dataloader
    dataloader_train = DataLoader(transformed_dataset_train["train"], batch_size=batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=8)
    dataloader_val = DataLoader(transformed_dataset_val["test"], batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader_train, dataloader_val


def fit_shape(t: torch.Tensor, x: torch.Tensor):
    return t.reshape(x.shape[0], *((1,) * (len(x.shape) - 1))).to(x.device)

def train_scorebased_model(model, criterion, optimizer, num_epochs, dataloader_train, noise_scheduler, device):
    liveloss = PlotLosses()
    model.to(device)
    global_step = 0
    uniform = torch.distributions.uniform.Uniform(0., 1.)

    for epoch in range(num_epochs):
        logs_liveplot = {}

        model.train()

        total_loss = 0.
        for step, batch in enumerate(dataloader_train):
            timesteps = uniform.sample((batch.shape[0],)).to(device)
            true_noise = torch.randn_like(batch)

            noisy_batch = noise_scheduler.q_sample(batch, timesteps, true_noise)

            predicted_noise = model(noisy_batch.to(device), timesteps.to(device))

            loss = criterion(predicted_noise, true_noise.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.detach().item()

        total_loss /= step
        logs_liveplot["loss"] = total_loss

        global_step += 1

        liveloss.update(logs_liveplot)
        liveloss.send()


def summarize(steps):
    s = torch.zeros((steps[0].shape[0], len(steps)))
    for i, step in enumerate(steps):
        s[:, len(steps) - i - 1] = step.squeeze()
    return s


def plot_line(y, ax, ax_left, ax_right, left_dist):
    ax_left.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xticks([0, 3.5, 7], [0.0, 0.5, 1.0])

    x = np.linspace(0, 7, y.shape[1])

    ax_max = max(y.max(), 4)
    ax_min = min(y.min(), -4)

    n = y.shape[0]
    for i in range(n):
        ax.plot(x, y[i, :], color='b')

    ax_left.plot(left_dist.log_prob(torch.linspace(ax_min, ax_max, 100)).exp(), np.linspace(ax_min, ax_max, 100),
                 color='b')

    n = torch.distributions.normal.Normal(loc=0, scale=1)
    ax_right.plot(n.log_prob(torch.linspace(ax_min, ax_max, 100)).exp(), np.linspace(ax_min, ax_max, 100), color='b')


def plot_1d_trajectories(steps, gmm):
    fig = plt.figure(layout='constrained')
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set(aspect=1)
    ax_left = ax.inset_axes([-0.3, 0, 0.25, 1], sharey=ax)
    ax_right = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    plot_line(summarize(steps).numpy(), ax, ax_left, ax_right, gmm)
    plt.show()


def plot_fashion(dataloader: DataLoader, output_dir: str, img_name="fashion_gridview.png"):
    images = next(iter(dataloader))['pixel_values'][:100]
    fig, axs = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(images[i * 10 + j].squeeze(), cmap="gray")
            axs[i, j].axis("off")
    plt.savefig(os.path.join(output_dir, img_name))
    plt.show()


def compare_beta_schedules(linear_schedule, improved_schedule):
    T = 1000

    linear_betas = linear_schedule(num_timesteps=T)
    alphas_l = 1. - linear_betas
    alphas_bar_linear = torch.cumprod(alphas_l, axis=0)

    improved_betas = improved_schedule(T)
    alphas_i = 1. - improved_betas
    alphas_bar_improved = torch.cumprod(alphas_i, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(np.arange(T), alphas_bar_linear.detach())
    ax1.plot(np.arange(T), alphas_bar_improved.detach())
    ax1.set_ylabel(r'$\bar{\alpha}_t$', fontsize=16)
    ax1.set_xlabel('t', fontsize=16)
    ax1.set_title(r'$\beta_{\rm start} = 0.0001$, $\beta_{\rm end} = 0.02$, and $T = 1000$')
    ax1.legend(['Linear Beta Schedule', 'Improved Beta Schedule'])

    T = 300

    linear_betas = linear_schedule(num_timesteps=T)
    alphas_l = 1. - linear_betas
    alphas_bar_linear = torch.cumprod(alphas_l, axis=0)

    improved_betas = improved_schedule(T)
    alphas_i = 1. - improved_betas
    alphas_bar_improved = torch.cumprod(alphas_i, axis=0)

    ax2.plot(np.arange(T), alphas_bar_linear.detach())
    ax2.plot(np.arange(T), alphas_bar_improved.detach())
    ax2.set_ylabel(r'$\bar{\alpha}_t$', fontsize=16)
    ax2.set_xlabel('t', fontsize=16)
    ax2.set_title(r'$\beta_{\rm start} = 0.0001$, $\beta_{\rm end} = 0.02$, and $T = 300$')
    ax2.legend(['Linear Beta Schedule', 'Improved Beta Schedule'])

    plt.show()


class DiffusionModel(pl.LightningModule):

    def __init__(self, model: nn.Module, loss_f: Callable, noise_scheduler):
        super().__init__()
        self.model = model
        self.loss_f = loss_f
        self.noise_scheduler = noise_scheduler

        # Default Initialization
        self.train_loss = 0.
        self.val_loss = 0.
        self.i_batch_train = 0
        self.i_batch_val = 0
        self.i_epoch = 0

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_noise = self.model(x_noisy, ts)

        loss = self.loss_f(noise, predicted_noise)
        self.log("train_loss", loss)
        self.train_loss += loss.detach().cpu().item()
        self.i_batch_train += 1
        return loss

    def on_train_epoch_end(self):
        print(' {}. Train Loss: {}'.format(self.i_epoch, self.train_loss / self.i_batch_train))
        self.train_loss = 0.
        self.i_batch_train = 0
        self.i_epoch += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)

        rng_state = torch.get_rng_state()
        torch.manual_seed(self.i_batch_val)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x)
        torch.set_rng_state(rng_state)

        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_noise = self.model(x_noisy, ts)

        loss = self.loss_f(noise, predicted_noise)
        self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        return loss

    def on_validation_epoch_end(self):
        print(' {}. Validation Loss: {}'.format(self.i_epoch, self.val_loss / self.i_batch_val))
        self.val_loss = 0.
        self.i_batch_val = 0


class DiffusionModelConditional(DiffusionModel):

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)
        y = batch['label'].to(self.device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_noise = self.model(x_noisy, ts, y)

        loss = self.loss_f(noise, predicted_noise)
        self.log("train_loss", loss)
        self.train_loss += loss.detach().cpu().item()
        self.i_batch_train += 1
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)
        y = batch['label'].to(self.device)

        rng_state = torch.get_rng_state()
        torch.manual_seed(self.i_batch_val)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x)
        torch.set_rng_state(rng_state)

        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_noise = self.model(x_noisy, ts, y)

        loss = self.loss_f(noise, predicted_noise)
        self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        return loss
