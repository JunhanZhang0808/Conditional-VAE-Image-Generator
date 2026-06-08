from pathlib import Path
import torch

from app.models import ConditionalVAE
from app.utils.text import Vocab


def load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    vocab = Vocab.from_dict(ckpt["vocab"])
    model = ConditionalVAE(
        image_size=config["image_size"],
        latent_dim=config["latent_dim"],
        vocab_size=len(vocab.token_to_id),
        text_dim=config["text_dim"],
        pad_id=vocab.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, config


def best_checkpoint(out_dir: str) -> str:
    path = Path(out_dir) / "checkpoints" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return str(path)
