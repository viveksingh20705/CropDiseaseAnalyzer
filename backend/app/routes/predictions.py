from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from bson import ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.database import get_database
from app.dependencies import get_current_user
from app.schemas.all_schemas import PredictionResponse
from app.services.ml_service import analyze_image

router = APIRouter(prefix="/api/predictions", tags=["predictions"])
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


def to_response(doc: dict) -> PredictionResponse:
    return PredictionResponse(
        id=str(doc["_id"]),
        crop=doc["crop"],
        disease=doc["disease"],
        class_label=doc["class_label"],
        is_healthy=doc["is_healthy"],
        severity=doc["severity"],
        confidence=doc["confidence"],
        top3=doc["top3"],
        symptoms=doc["symptoms"],
        treatment=doc["treatment"],
        prevention=doc["prevention"],
        image_url=doc["image_url"],
        created_at=doc["created_at"],
    )


@router.post("/analyze", response_model=PredictionResponse)
async def analyze(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPEG, PNG, or WebP leaf image")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "leaf.jpg").suffix.lower() or ".jpg"
    stored_name = f"{uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / stored_name
    content = await file.read()
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be smaller than 8 MB")
    destination.write_bytes(content)

    try:
        result = analyze_image(str(destination))
    except FileNotFoundError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    document = {
        "user_id": str(current_user["_id"]),
        "filename": stored_name,
        "image_url": f"/uploads/{stored_name}",
        "created_at": datetime.now(timezone.utc),
        **result,
    }
    database = get_database()
    inserted = await database.predictions.insert_one(document)
    document["_id"] = inserted.inserted_id
    return to_response(document)


@router.get("/history", response_model=list[PredictionResponse])
async def history(current_user: dict = Depends(get_current_user)):
    database = get_database()
    query = {} if current_user.get("role") == "admin" else {"user_id": str(current_user["_id"])}
    cursor = database.predictions.find(query).sort("created_at", -1)
    return [to_response(doc) async for doc in cursor]


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prediction(prediction_id: str, current_user: dict = Depends(get_current_user)):
    database = get_database()
    try:
        object_id = ObjectId(prediction_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Prediction not found") from exc
    document = await database.predictions.find_one({"_id": object_id})
    if not document:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if current_user.get("role") != "admin" and document.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="You can only delete your own analyses")
    await database.predictions.delete_one({"_id": object_id})
    image_path = UPLOAD_DIR / document.get("filename", "")
    if image_path.exists():
        image_path.unlink()
    return None
