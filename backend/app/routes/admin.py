from fastapi import APIRouter, Depends

from app.database import get_database
from app.dependencies import get_admin_user

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
async def admin_stats(admin_user: dict = Depends(get_admin_user)):
    database = get_database()
    users = await database.users.count_documents({})
    farmers = await database.users.count_documents({"role": "farmer"})
    admins = await database.users.count_documents({"role": "admin"})
    predictions = await database.predictions.count_documents({})
    diseased = await database.predictions.count_documents({"is_healthy": False})
    healthy = await database.predictions.count_documents({"is_healthy": True})
    crops = await database.crops.count_documents({})
    diseases = await database.diseases.count_documents({})
    latest = await database.predictions.find().sort("created_at", -1).limit(8).to_list(8)
    return {
        "users": users,
        "farmers": farmers,
        "admins": admins,
        "predictions": predictions,
        "diseased": diseased,
        "healthy": healthy,
        "crops": crops,
        "diseases": diseases,
        "recent_predictions": [
            {
                "id": str(item["_id"]),
                "crop": item.get("crop"),
                "disease": item.get("disease"),
                "confidence": item.get("confidence"),
                "created_at": item.get("created_at"),
            }
            for item in latest
        ],
    }
