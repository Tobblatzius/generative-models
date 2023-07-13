import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def generate_samples(images, model, device="cpu"):
    with torch.no_grad():
        images = images.to(device)
        model.to(device)
        x_tilde, latents_og_grad, z, perplexity = model(images)
    return x_tilde


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            return

def tensor_to_pil_image(tensor):
    return Image.fromarray(tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(
        self,
        mask_type,
        dim,
        kernel,
        residual=True,
        n_classes=10,
        conditional=False,
        in_dim=None,
    ):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        in_dim = dim if in_dim is None else in_dim

        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)
        if not conditional:
            self.class_cond_embedding.weight.data.zero_()
            self.class_cond_embedding.weight.requires_grad = False

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(in_dim, dim * 2, kernel_shp, 1, padding_shp)

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(in_dim, dim * 2, kernel_shp, 1, padding_shp)

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == "A":
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, : x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, : x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(
        self,
        codebook_size=256,
        hidden_dim=32,
        n_layers=15,
        n_classes=10,
        conditional=False,
    ):
        super().__init__()
        self.dim = hidden_dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(codebook_size, hidden_dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = "A" if i == 0 else "B"
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            in_dim = hidden_dim if i > 0 else hidden_dim

            self.layers.append(
                GatedMaskedConv2d(
                    mask_type,
                    hidden_dim,
                    kernel,
                    residual,
                    n_classes,
                    conditional,
                    in_dim,
                )
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, codebook_size, 1),
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.view(shp).permute(0, 3, 1, 2).float()  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(7, 7), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros((batch_size, *shape), dtype=torch.int64, device=param.device)

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return x
