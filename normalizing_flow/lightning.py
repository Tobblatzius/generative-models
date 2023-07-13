import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.utils.data as data
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import show_imgs

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "data/"

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved_models/"

# Setting the seed
pl.seed_everything(42)

# torch.backends.cudnn.benchmark = True


class LightningFlow(pl.LightningModule):
    def __init__(self, flow: "NormalizingFlow", example_input, import_samples=8):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flow = flow
        self.import_samples = import_samples
        self.example_input_array = example_input[None, ...]

    def _get_likelihood(self, imgs, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        log_prob = self.flow.log_prob(imgs)
        nll = -log_prob
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_prob

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch[0])
        self.log("train_bpd", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch[0])
        self.log("val_bpd", loss, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        # Perform importance sampling during testing => estimate likelihood M times for each image
        samples = []
        for _ in range(self.import_samples):
            img_ll = self._get_likelihood(batch[0], return_ll=True)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)

        # To average the probabilities, we need to go from log-space to exp, and back to log.
        # Logsumexp provides us a stable implementation for this
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        # Calculate final bpd
        bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
        bpd = bpd.mean()

        self.log("test_bpd", bpd)

    def on_train_epoch_end(self):
        # After each epoch, we can plot some faces to the notebook
        samples = self.flow.sample(shape=[8, 8, 7, 7])
        show_imgs(samples.cpu())


def train_flow_lightning(
    flow,
    model_name,
    max_epochs,
    device,
    train_set,
    val_loader,
    test_loader,
    batch_size=128,
    **trainer_kwargs
):
    # Create a PyTorch Lightning trainer
    model = LightningFlow(flow, train_set[0][0]).to(device)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=1,
        **trainer_kwargs,
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    train_data_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
    result = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        ckpt = torch.load(pretrained_filename, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        result = ckpt.get("result", None)
    else:
        print("Start training", model_name)
        trainer.fit(model, train_data_loader, val_loader)

    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(model, val_loader, verbose=False)
        start_time = time.time()
        test_result = trainer.test(model, test_loader, verbose=False)
        duration = time.time() - start_time
        result = {
            "test": test_result,
            "val": val_result,
            "time": duration / len(test_loader) / model.import_samples,
        }

    return model, result, trainer
