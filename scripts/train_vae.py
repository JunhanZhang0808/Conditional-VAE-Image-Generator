import argparse

from app.services.training_service import train_vae


def parse_args():
    parser = argparse.ArgumentParser(description="Train a prompt-conditioned VAE on your own image folder.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/vae/default_run")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--text_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--max_text_len", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_vae(**vars(args))
    print("Training finished.")
    print(result)


if __name__ == "__main__":
    main()
