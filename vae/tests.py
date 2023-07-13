import torch
import numpy as np
from vq_vae import ResBlock, VectorQuantizedEmbedding, VectorQuantizedVAEDecoder, VectorQuantizedVAEEncoder, VectorQuantizedVAE, vqvae_loss

def test_resblock():
    res_block = ResBlock(3, 4, 3)
    x = torch.randn(2, 3, 4, 4)
    y = res_block(x)
    assert y.shape == torch.Size([2, 3, 4, 4])
    print("Success!")

def test_vqvae_loss(func):
    x = torch.randn(2, 3, 4, 4)
    x_recon = torch.randn(2, 3, 4, 4)
    quantized_latents = torch.randn(2, 3, 4, 4)
    latents = torch.randn(2, 3, 4, 4)
    loss, recon_loss, vq_loss, commit_loss = func(
        x, x_recon, quantized_latents, latents, commit_loss_weight=0.25
    )
    assert loss.shape == torch.Size([])
    assert recon_loss.shape == torch.Size([])
    assert vq_loss.shape == torch.Size([])
    assert commit_loss.shape == torch.Size([])
    assert loss == recon_loss + vq_loss + 0.25 * commit_loss

    # Check that the loss is correct
    recon_loss_np = np.mean((x_recon.numpy() - x.numpy()) ** 2)
    commit_loss_np = np.mean((quantized_latents.numpy() - latents.numpy()) ** 2)
    vq_loss_np = np.mean((quantized_latents.numpy() - latents.numpy()) ** 2)
    loss_np = recon_loss + vq_loss + 0.25 * commit_loss
    assert np.allclose(loss.numpy(), loss_np)
    
    print("Success!")

def test_vqvae_encoder_decoder():
    encoder = VectorQuantizedVAEEncoder(3, 64, 32)
    decoder = VectorQuantizedVAEDecoder(64, 64, 32, 3)
    x = torch.randn(2, 3, 32, 32)
    y = decoder(encoder(x))
    assert y.shape == torch.Size([2, 3, 32, 32])
    print("Success!")

def test_vqvae_codebook():
    codebook = VectorQuantizedEmbedding(10, 64)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y, latents_input_grad, perplexity, encoding_indices = codebook(x)
    assert y.shape == torch.Size([2, 64, 32, 32])
    assert latents_input_grad.shape == torch.Size([2, 64, 32, 32])
    assert perplexity.shape == torch.Size([])
    assert encoding_indices.shape == torch.Size([2, 32, 32])
    assert encoding_indices.dtype == torch.int64

    print("Success!")

def test_vqvae():
    vqvae = VectorQuantizedVAE(3, 64, 512, 64, 32)
    x = torch.randn(2, 3, 28, 28, requires_grad=True)
    y, latents_input_grad, z, perplexity = vqvae(x)
    assert y.shape == torch.Size([2, 3, 28, 28])
    assert latents_input_grad.shape == torch.Size([2, 64, 7, 7])
    assert z.shape == torch.Size([2, 64, 7, 7])
    assert perplexity.shape == torch.Size([])
    print("Success!")

test_vqvae_loss(vqvae_loss)
test_resblock()
test_vqvae_encoder_decoder()
test_vqvae_codebook()
test_vqvae()