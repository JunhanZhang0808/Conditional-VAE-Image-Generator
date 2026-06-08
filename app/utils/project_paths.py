from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GENERATED_DIR = OUTPUTS_DIR / "generated"
VAE_OUTPUT_DIR = OUTPUTS_DIR / "vae"
