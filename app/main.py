from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Prompt Conditional VAE System", version="0.2.0")
app.include_router(router, prefix="/api")
