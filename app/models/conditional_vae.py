from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConditionEncoder(nn.Module):
    def __init__(self, vocab_size: int, text_dim: int = 128, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=pad_id)
        self.pad_id = pad_id

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L]
        emb = self.embedding(token_ids)  # [B, L, D]
        mask = (token_ids != self.pad_id).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count


class ConditionalVAE(nn.Module):
    """
    轻量级文本条件 VAE。

    注意：这里的 prompt 只是训练 caption 的弱条件，不具备 Stable Diffusion 级别的语义理解能力。
    """

    def __init__(
        self,
        image_size: int = 64,
        latent_dim: int = 128,
        vocab_size: int = 1000,
        text_dim: int = 128,
        pad_id: int = 0,
    ):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16, e.g. 64, 128, 256")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.start_size = image_size // 16

        self.text_encoder = TextConditionEncoder(vocab_size, text_dim, pad_id)

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        flat_dim = 256 * self.start_size * self.start_size
        self.fc_hidden = nn.Sequential(
            nn.Linear(flat_dim + text_dim, 512),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, flat_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self.text_encoder(token_ids)
        h = self.encoder_cnn(x)
        h = torch.flatten(h, start_dim=1)
        h = torch.cat([h, cond], dim=1)
        h = self.fc_hidden(h)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        cond = self.text_encoder(token_ids)
        h = torch.cat([z, cond], dim=1)
        h = self.decoder_fc(h)
        h = h.view(-1, 256, self.start_size, self.start_size)
        return self.decoder_cnn(h)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor):
        mu, logvar = self.encode(x, token_ids)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, token_ids)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta: float = 0.001, recon_type: str = "mse"):
    if recon_type == "bce":
        recon = F.binary_cross_entropy(recon_x, x, reduction="mean")
    else:
        recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon.detach(), kl.detach()
