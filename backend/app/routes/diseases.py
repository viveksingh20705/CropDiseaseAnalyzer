from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_database
from app.dependencies import get_admin_user, get_current_user
from app.schemas.all_schemas import DiseaseSchema

router = APIRouter(prefix="/api/diseases", tags=["diseases"])


def slugify(crop: str, name: str) -> str:
    return f"{crop}___{name}".replace(" ", "_")


def serialize_disease(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "name": doc["name"],
        "crop": doc["crop"],
        "slug": doc.get("slug"),
        "severity": doc.get("severity", "Moderate"),
        "symptoms": doc.get("symptoms", []),
        "treatment": doc.get("treatment", []),
        "prevention": doc.get("prevention", []),
        "description": doc.get("description"),
    }


@router.get("")
async def list_diseases(current_user: dict = Depends(get_current_user)):
    database = get_database()
    return [serialize_disease(doc) async for doc in database.diseases.find().sort("crop", 1)]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_disease(payload: DiseaseSchema, admin_user: dict = Depends(get_admin_user)):
    database = get_database()
    document = payload.model_dump()
    document["slug"] = payload.slug or slugify(payload.crop, payload.name)
    existing = await database.diseases.find_one({"slug": document["slug"]})
    if existing:
        raise HTTPException(status_code=400, detail="Disease already exists")
    result = await database.diseases.insert_one(document)
    document["_id"] = result.inserted_id
    return serialize_disease(document)
