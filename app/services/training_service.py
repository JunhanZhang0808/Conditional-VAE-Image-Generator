from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from app.models import ConditionalVAE, vae_loss
from app.utils.dataset import ImageCaptionDataset, collect_captions
from app.utils.text import Vocab
from app.utils.image_io import ensure_dir, save_image_grid, set_seed


def default_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_vae(
    data_dir: str,
    out_dir: str = "outputs/vae/default_run",
    image_size: int = 64,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    latent_dim: int = 128,
    text_dim: int = 128,
    beta: float = 0.001,
    max_text_len: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    device: str | None = None,
    save_every: int = 10,
) -> Dict[str, Any]:
    device = default_device(device)
    set_seed(seed)

    out_path = Path(out_dir)
    ensure_dir(out_path)
    ensure_dir(out_path / "samples")
    ensure_dir(out_path / "checkpoints")

    captions = collect_captions(data_dir)
    vocab = Vocab.build(captions, min_freq=1, max_size=5000)
    vocab.save(out_path / "vocab.json")

    dataset = ImageCaptionDataset(data_dir, vocab, image_size, max_text_len)
    val_size = max(1, int(0.1 * len(dataset))) if len(dataset) >= 10 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None

    model = ConditionalVAE(
        image_size=image_size,
        latent_dim=latent_dim,
        vocab_size=len(vocab.token_to_id),
        text_dim=text_dim,
        pad_id=vocab.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    config = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "latent_dim": latent_dim,
        "text_dim": text_dim,
        "beta": beta,
        "max_text_len": max_text_len,
        "num_workers": num_workers,
        "seed": seed,
        "device": device,
        "vocab_size": len(vocab.token_to_id),
        "pad_id": vocab.pad_id,
    }

    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, token_ids, _ in pbar:
            images = images.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)

            recon, mu, logvar = model(images, token_ids)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta=beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", kl=f"{kl_loss.item():.4f}")

        train_loss = running_total / max(1, len(train_loader))

        val_loss = None
        if val_loader:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for images, token_ids, _ in val_loader:
                    images = images.to(device)
                    token_ids = token_ids.to(device)
                    recon, mu, logvar = model(images, token_ids)
                    loss, _, _ = vae_loss(recon, images, mu, logvar, beta=beta)
                    total += loss.item()
            val_loss = total / max(1, len(val_loader))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}" + (f", val_loss={val_loss:.6f}" if val_loss is not None else ""))

        if epoch == 1 or epoch % save_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                images, token_ids, _ = next(iter(train_loader))
                images = images.to(device)[:16]
                token_ids = token_ids.to(device)[:16]
                recon, _, _ = model(images, token_ids)
                comparison = torch.cat([images.cpu(), recon.cpu()], dim=0)
                save_image_grid(comparison, out_path / "samples" / f"recon_epoch_{epoch:04d}.png", nrow=8)

            ckpt = {
                "model_state": model.state_dict(),
                "config": config,
                "vocab": vocab.to_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(ckpt, out_path / "checkpoints" / f"vae_epoch_{epoch:04d}.pt")
            torch.save(ckpt, out_path / "checkpoints" / "latest.pt")

            score = val_loss if val_loss is not None else train_loss
            if score < best_val:
                best_val = score
                torch.save(ckpt, out_path / "checkpoints" / "best.pt")

    return {
        "out_dir": str(out_path),
        "best_ckpt": str(out_path / "checkpoints" / "best.pt"),
        "latest_ckpt": str(out_path / "checkpoints" / "latest.pt"),
        "history": history,
        "num_images": len(dataset),
        "vocab_size": len(vocab.token_to_id),
        "device": device,
    }
