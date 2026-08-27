from fastapi import APIRouter, Depends

from app.database import get_database
from app.dependencies import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
async def stats(current_user: dict = Depends(get_current_user)):
    database = get_database()
    match = {} if current_user.get("role") == "admin" else {"user_id": str(current_user["_id"])}
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "healthy_count": {"$sum": {"$cond": ["$is_healthy", 1, 0]}},
                "diseased_count": {"$sum": {"$cond": ["$is_healthy", 0, 1]}},
                "avg_confidence": {"$avg": "$confidence"},
            }
        },
    ]
    grouped = await database.predictions.aggregate(pipeline).to_list(1)
    totals = grouped[0] if grouped else {
        "total_predictions": 0,
        "healthy_count": 0,
        "diseased_count": 0,
        "avg_confidence": 0,
    }

    disease_rows = await database.predictions.aggregate(
        [
            {"$match": match},
            {"$group": {"_id": "$disease", "value": {"$sum": 1}}},
            {"$sort": {"value": -1}},
        ]
    ).to_list(20)
    crop_rows = await database.predictions.aggregate(
        [
            {"$match": match},
            {"$group": {"_id": "$crop", "value": {"$sum": 1}}},
            {"$sort": {"value": -1}},
        ]
    ).to_list(20)

    return {
        "total_predictions": int(totals.get("total_predictions") or 0),
        "healthy_count": int(totals.get("healthy_count") or 0),
        "diseased_count": int(totals.get("diseased_count") or 0),
        "avg_confidence": round(float(totals.get("avg_confidence") or 0), 2),
        "disease_distribution": [{"name": row["_id"], "value": row["value"]} for row in disease_rows],
        "crop_distribution": [{"name": row["_id"], "value": row["value"]} for row in crop_rows],
    }
