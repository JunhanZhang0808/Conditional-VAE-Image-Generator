import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from torchvision import datasets

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/","_").lower()

def save_image(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    img.save(path, quality=95)

def export_cifar10(root: str, out_dir: str, split: str):
    train = split == "train"
    ds = datasets.CIFAR10(root=root, train=train, download=True)

    class_names = ds.classes
    counters = {name: 0 for name in class_names}

    for img, label in tqdm(ds, desc=f"Export CIFAR10 {split}"):
        cls_name = class_names[label]
        cls = safe_name(cls_name)
        idx = counters[cls_name]
        counters[cls_name] += 1

        save_path = Path(out_dir) / cls / f"{idx:06d}.jpg"
        save_image(img, save_path)

def export_pet(root: str, out_dir: str, split: str):
    ds = datasets.OxfordIIITPet(
        root=root,
        split="trainval" if split == "train" else "test",
        target_types="category",
        download=True,
    )

    class_names = ds.classes
    counters = {name: 0 for name in class_names}

    for img, label in tqdm(ds, desc=f"Export Oxford-IIIT Pet {split}"):
        cls_name = class_names[label]
        cls = safe_name(cls_name)
        idx = counters[cls_name]
        counters[cls_name] += 1

        save_path = Path(out_dir) / cls / f"{idx:06d}.jpg"
        save_image(img, save_path)

def export_flowers(root: str, out_dir: str, split: str):
    ds = datasets.Flowers102(
        root=root,
        split=split,
        download=True,
    )

    counters = {}

    for img, label in tqdm(ds, desc=f"Export Flowers102 {split}"):
        cls = f"flower_{label + 1:03d}"
        counters.setdefault(cls, 0)
        idx = counters[cls]
        counters[cls] += 1

        save_path = Path(out_dir) / cls / f"{idx:06d}.jpg"
        save_image(img, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "pet", "flowers"])
    parser.add_argument("--root", type=str, default="datasets/raw")
    parser.add_argument("--out_dir", type=str, default="datasets/my_images")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "val"])
    args = parser.parse_args()

    if args.dataset == "cifar10":
        export_cifar10(args.root, args.out_dir, args.split)
    elif args.dataset == "pet":
        export_pet(args.root, args.out_dir, args.split)
    elif args.dataset == "flowers":
        export_flowers(args.root, args.out_dir, args.split)

    print(f"Done. Exported to: {args.out_dir}")


if __name__ == "__main__":
    main()