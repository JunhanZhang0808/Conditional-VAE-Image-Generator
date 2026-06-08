from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from app.services.checkpoint_service import load_checkpoint
from app.utils.image_io import ensure_dir, save_image_grid, set_seed


def default_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def generate_images(
    ckpt: str,
    prompt: str,
    out_dir: str = "outputs/generated",
    num_images: int = 4,
    seed: int = 123,
    temperature: float = 1.0,
    device: str | None = None,
) -> Tuple[List[str], str]:
    device = default_device(device)
    set_seed(seed)
    ensure_dir(out_dir)
    model, vocab, config = load_checkpoint(ckpt, device)

    token_ids = torch.tensor(
        [vocab.encode(prompt, config["max_text_len"])] * num_images,
        dtype=torch.long,
    ).to(device)

    with torch.no_grad():
        z = torch.randn(num_images, config["latent_dim"], device=device) * temperature
        images = model.decode(z, token_ids).detach().cpu().clamp(0, 1)

    out = Path(out_dir)
    image_paths = []
    for i, img in enumerate(images):
        path = out / f"sample_{i:03d}.png"
        save_image(img, path)
        image_paths.append(str(path))
    grid_path = out / "grid.png"
    save_image_grid(images, grid_path, nrow=min(4, num_images))
    return image_paths, str(grid_path)


def reconstruct_image(
    ckpt: str,
    image_path: str,
    prompt: str = "image",
    out_dir: str = "outputs/generated/reconstruct",
    seed: int = 123,
    device: str | None = None,
) -> str:
    device = default_device(device)
    set_seed(seed)
    ensure_dir(out_dir)
    model, vocab, config = load_checkpoint(ckpt, device)

    tfm = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    token_ids = torch.tensor([vocab.encode(prompt, config["max_text_len"])], dtype=torch.long).to(device)

    with torch.no_grad():
        recon, _, _ = model(x, token_ids)

    pair = torch.cat([x.cpu(), recon.cpu()], dim=0)
    out_path = Path(out_dir) / "reconstruct.png"
    save_image_grid(pair, out_path, nrow=2)
    return str(out_path)
