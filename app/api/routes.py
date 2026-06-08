from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File

from app.models.schemas import TrainRequest, GenerateRequest, ReconstructRequest
from app.services.training_service import train_vae
from app.services.generation_service import generate_images, reconstruct_image
from app.utils.image_io import ensure_dir

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "project": "vae_prompt_project_v2"}


@router.post("/upload_dataset")
async def upload_dataset(files: List[UploadFile] = File(...), save_dir: str = "datasets/my_images"):
    ensure_dir(save_dir)
    saved = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".txt"}:
            continue
        target = Path(save_dir) / f.filename
        content = await f.read()
        target.write_bytes(content)
        saved.append(str(target))
    return {"saved": saved, "count": len(saved)}


@router.post("/train")
def train(req: TrainRequest):
    # 注意：API 方式会阻塞直到训练完成。正式系统应改成任务队列。
    result = train_vae(**req.model_dump())
    return result


@router.post("/generate")
def generate(req: GenerateRequest):
    image_paths, grid_path = generate_images(**req.model_dump())
    return {"image_paths": image_paths, "grid_path": grid_path}


@router.post("/reconstruct")
def reconstruct(req: ReconstructRequest):
    path = reconstruct_image(**req.model_dump())
    return {"reconstruct_path": path}
