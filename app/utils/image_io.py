from pathlib import Path
import random
import numpy as np
import torch
from torchvision.utils import save_image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_image_grid(tensor, path, nrow: int = 8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor, path, nrow=nrow, normalize=False)
    return str(path)
