import argparse
from pathlib import Path

import gradio as gr

from app.services.generation_service import generate_images
from app.utils.image_io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for prompt-conditioned VAE.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/generated/gradio")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    def run_generate(prompt, num_images, seed, temperature):
        run_dir = Path(args.out_dir) / f"seed_{int(seed)}"
        image_paths, grid_path = generate_images(
            ckpt=args.ckpt,
            prompt=prompt,
            out_dir=str(run_dir),
            num_images=int(num_images),
            seed=int(seed),
            temperature=float(temperature),
            device=args.device,
        )
        return image_paths, grid_path

    with gr.Blocks() as demo:
        gr.Markdown("# Prompt Conditional VAE Demo\n纯 VAE/CVAE 教学版：prompt 只对训练集 caption 里的词有弱控制作用。")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="image")
                num_images = gr.Slider(1, 16, value=4, step=1, label="Number of images")
                seed = gr.Number(value=123, precision=0, label="Seed")
                temperature = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="Latent temperature")
                btn = gr.Button("Generate")
            with gr.Column():
                gallery = gr.Gallery(label="Generated Images", columns=4)
                grid = gr.Image(label="Grid")
        btn.click(run_generate, inputs=[prompt, num_images, seed, temperature], outputs=[gallery, grid])

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
