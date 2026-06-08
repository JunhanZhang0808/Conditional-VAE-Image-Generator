from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    data_dir: str = Field(default="datasets/my_images")
    out_dir: str = Field(default="outputs/vae/api_run")
    image_size: int = 64
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    latent_dim: int = 128
    text_dim: int = 128
    beta: float = 0.001
    max_text_len: int = 32
    num_workers: int = 2
    seed: int = 42


class GenerateRequest(BaseModel):
    ckpt: str
    prompt: str = "image"
    out_dir: str = "outputs/generated/api"
    num_images: int = 4
    seed: int = 123
    temperature: float = 1.0


class ReconstructRequest(BaseModel):
    ckpt: str
    image_path: str
    prompt: str = "image"
    out_dir: str = "outputs/generated/reconstruct"
    seed: int = 123
