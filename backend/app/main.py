from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import close_mongo_connection, connect_to_mongo
from app.routes import admin, auth, crops, dashboard, diseases, predictions

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app = FastAPI(
    title="Crop Disease Analyzer API",
    version="1.0.0",
    description="College-level API for crop leaf disease diagnosis with MongoDB persistence.",
    lifespan=lifespan,
)

frontend_origin = settings.frontend_origin.strip().rstrip("/")
if frontend_origin and "://" not in frontend_origin:
    frontend_origin = f"https://{frontend_origin}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_origin,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://viveksingh20705.github.io",
        "https://crop-disease-analyzer.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)
app.include_router(crops.router)
app.include_router(diseases.router)
app.include_router(admin.router)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "crop-disease-analyzer"}
