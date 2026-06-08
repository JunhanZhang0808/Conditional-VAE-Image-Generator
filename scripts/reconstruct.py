import argparse

from app.services.generation_service import reconstruct_image


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct one image with a trained VAE.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="image")
    parser.add_argument("--out_dir", type=str, default="outputs/generated/reconstruct")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    path = reconstruct_image(**vars(args))
    print(f"Saved reconstruction to {path}")


if __name__ == "__main__":
    main()
