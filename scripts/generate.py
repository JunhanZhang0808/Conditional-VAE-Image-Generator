import argparse

from app.services.generation_service import generate_images


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with a trained prompt-conditioned VAE.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="image")
    parser.add_argument("--out_dir", type=str, default="outputs/generated")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths, grid_path = generate_images(**vars(args))
    print(f"Saved images: {image_paths}")
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
