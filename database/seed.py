"""Seed MongoDB with an admin account, crop catalog, and disease library."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
load_dotenv(BACKEND / ".env")
sys.path.insert(0, str(BACKEND))

from app.config import settings  # noqa: E402

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

CROPS = [
    {
        "name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Warm-season fruiting vegetable widely grown in open fields and polyhouses.",
        "growing_season": "Spring–Summer",
    },
    {
        "name": "Potato",
        "scientific_name": "Solanum tuberosum",
        "description": "Tuber crop sensitive to late blight during cool, wet weather.",
        "growing_season": "Winter–Spring",
    },
    {
        "name": "Corn",
        "scientific_name": "Zea mays",
        "description": "Cereal grown for grain and fodder; rust and leaf blights reduce yield.",
        "growing_season": "Kharif / Summer",
    },
    {
        "name": "Apple",
        "scientific_name": "Malus domestica",
        "description": "Temperate orchard crop. Scab and rust are the main foliar constraints.",
        "growing_season": "Spring bloom, autumn harvest",
    },
    {
        "name": "Grape",
        "scientific_name": "Vitis vinifera",
        "description": "Vine crop requiring open canopies to limit fungal fruit rots.",
        "growing_season": "Spring–Monsoon",
    },
    {
        "name": "Pepper",
        "scientific_name": "Capsicum annuum",
        "description": "Bell and chili peppers; bacterial spot is common in humid nurseries.",
        "growing_season": "Warm season",
    },
    {
        "name": "Cherry",
        "scientific_name": "Prunus avium",
        "description": "Stone fruit. Powdery mildew thrives in dense, shaded canopies.",
        "growing_season": "Spring",
    },
    {
        "name": "Strawberry",
        "scientific_name": "Fragaria × ananassa",
        "description": "Perennial berry. Leaf scorch and fruit rots increase after rain.",
        "growing_season": "Cool season",
    },
    {
        "name": "Squash",
        "scientific_name": "Cucurbita pepo",
        "description": "Cucurbit vegetable highly susceptible to powdery mildew.",
        "growing_season": "Summer",
    },
    {
        "name": "Soybean",
        "scientific_name": "Glycine max",
        "description": "Oilseed legume. Foliar diseases reduce seed fill in humid years.",
        "growing_season": "Kharif",
    },
]

DISEASES = [
    {
        "name": "Early Blight",
        "crop": "Tomato",
        "slug": "Tomato___Early_blight",
        "severity": "High",
        "description": "Alternaria-driven target-spot disease that starts on older leaves.",
        "symptoms": ["Concentric leaf lesions", "Yellow halos", "Lower-canopy defoliation"],
        "treatment": ["Remove infected leaves", "Apply labeled protectant fungicide"],
        "prevention": ["Mulch to stop soil splash", "Stake plants", "Rotate solanaceous crops"],
    },
    {
        "name": "Late Blight",
        "crop": "Potato",
        "slug": "Potato___Late_blight",
        "severity": "Critical",
        "description": "Phytophthora infestans outbreak that can destroy a canopy in days.",
        "symptoms": ["Water-soaked lesions", "White sporulation", "Tuber rot"],
        "treatment": ["Rogue outbreaks", "Use a systemic + protectant program"],
        "prevention": ["Certified seed", "Kill vines before harvest", "Dry storage"],
    },
    {
        "name": "Common Rust",
        "crop": "Corn",
        "slug": "Corn___Common_rust",
        "severity": "Moderate",
        "description": "Puccinia sorghi pustules that reduce photosynthetic area.",
        "symptoms": ["Cinnamon pustules on both leaf faces"],
        "treatment": ["Fungicide before tasseling if threshold is exceeded"],
        "prevention": ["Resistant hybrids", "Avoid late planting in rust belts"],
    },
    {
        "name": "Apple Scab",
        "crop": "Apple",
        "slug": "Apple___Apple_scab",
        "severity": "High",
        "description": "Venturia inaequalis infection of leaves and fruit.",
        "symptoms": ["Olive velvety spots", "Cracked fruit russet"],
        "treatment": ["Protectant sprays from green tip"],
        "prevention": ["Resistant cultivars", "Shred fallen leaves"],
    },
    {
        "name": "Black Rot",
        "crop": "Grape",
        "slug": "Grape___Black_rot",
        "severity": "High",
        "description": "Guignardia fruit rot that mummifies berries.",
        "symptoms": ["Tan leaf spots", "Shriveled black berries"],
        "treatment": ["Bloom-to-veraison fungicide program"],
        "prevention": ["Remove mummies", "Open the fruiting zone"],
    },
    {
        "name": "Bacterial Spot",
        "crop": "Pepper",
        "slug": "Pepper,_bell___Bacterial_spot",
        "severity": "High",
        "description": "Xanthomonas lesions on leaves and fruit.",
        "symptoms": ["Greasy specks", "Raised fruit scabs"],
        "treatment": ["Copper during wet weather"],
        "prevention": ["Pathogen-free seed", "Drip irrigation"],
    },
    {
        "name": "Powdery Mildew",
        "crop": "Cherry",
        "slug": "Cherry___Powdery_mildew",
        "severity": "Moderate",
        "description": "Podosphaera white growth on young shoots.",
        "symptoms": ["White powder", "Leaf curling"],
        "treatment": ["Sulfur or potassium bicarbonate"],
        "prevention": ["Open pruning", "Avoid excess nitrogen"],
    },
    {
        "name": "Leaf Scorch",
        "crop": "Strawberry",
        "slug": "Strawberry___Leaf_scorch",
        "severity": "Moderate",
        "description": "Diplocarpon foliar blight after long dew periods.",
        "symptoms": ["Purple spots coalescing into scorched patches"],
        "treatment": ["Mow and remove old leaves after harvest"],
        "prevention": ["Wider spacing", "Morning irrigation only"],
    },
]


async def seed() -> None:
    client = AsyncIOMotorClient(settings.mongodb_uri)
    database = client[settings.mongodb_db]

    admin_email = "admin@cropanalyzer.com"
    existing_admin = await database.users.find_one({"email": admin_email})
    if existing_admin:
        print("Admin user already exists — skipped.")
    else:
        await database.users.insert_one(
            {
                "name": "System Administrator",
                "email": admin_email,
                "password_hash": pwd_context.hash("Admin@123456"),
                "role": "admin",
                "created_at": datetime.now(timezone.utc),
            }
        )
        print("Created admin@cropanalyzer.com")

    for crop in CROPS:
        await database.crops.update_one({"name": crop["name"]}, {"$set": crop}, upsert=True)
    print(f"Upserted {len(CROPS)} crops")

    for disease in DISEASES:
        await database.diseases.update_one({"slug": disease["slug"]}, {"$set": disease}, upsert=True)
    print(f"Upserted {len(DISEASES)} diseases")

    await database.users.create_index("email", unique=True)
    client.close()
    print("Seed complete.")


if __name__ == "__main__":
    asyncio.run(seed())
