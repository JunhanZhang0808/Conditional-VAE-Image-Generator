import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create same-name .txt captions from parent folder names.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_dir)
    count = 0
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        txt = img.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            continue
        caption = img.parent.name.replace("_", " ") if img.parent != root else "image"
        txt.write_text(caption, encoding="utf-8")
        count += 1
    print(f"Created/updated {count} caption files.")


if __name__ == "__main__":
    main()
