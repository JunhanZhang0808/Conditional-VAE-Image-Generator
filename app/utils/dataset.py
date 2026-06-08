from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from app.utils.text import Vocab

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def caption_for_image(path: Path, root: Path) -> str:
    txt = path.with_suffix(".txt")
    if txt.exists():
        text = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    if path.parent != root:
        return path.parent.name.replace("_", " ")
    return "image"


class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir: str, vocab: Vocab, image_size: int = 64, max_text_len: int = 32):
        self.root = Path(data_dir)
        self.paths = find_images(data_dir)
        if not self.paths:
            raise RuntimeError(f"No images found in {data_dir}")
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        caption = caption_for_image(path, self.root)
        token_ids = torch.tensor(self.vocab.encode(caption, self.max_text_len), dtype=torch.long)
        return image, token_ids, caption


def collect_captions(data_dir: str) -> List[str]:
    root = Path(data_dir)
    paths = find_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found in {data_dir}")
    return [caption_for_image(p, root) for p in paths]
