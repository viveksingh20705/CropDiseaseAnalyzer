from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_database
from app.dependencies import get_admin_user, get_current_user
from app.schemas.all_schemas import CropSchema

router = APIRouter(prefix="/api/crops", tags=["crops"])


def serialize_crop(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "name": doc["name"],
        "scientific_name": doc.get("scientific_name"),
        "description": doc.get("description"),
        "growing_season": doc.get("growing_season"),
    }


@router.get("")
async def list_crops(current_user: dict = Depends(get_current_user)):
    database = get_database()
    crops = [serialize_crop(doc) async for doc in database.crops.find().sort("name", 1)]
    return crops


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_crop(payload: CropSchema, admin_user: dict = Depends(get_admin_user)):
    database = get_database()
    existing = await database.crops.find_one({"name": payload.name})
    if existing:
        raise HTTPException(status_code=400, detail="Crop already exists")
    document = payload.model_dump()
    result = await database.crops.insert_one(document)
    document["_id"] = result.inserted_id
    return serialize_crop(document)
