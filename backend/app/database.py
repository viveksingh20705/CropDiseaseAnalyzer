from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

client: AsyncIOMotorClient | None = None
db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo() -> None:
    global client, db
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]
    await db.users.create_index("email", unique=True)
    await db.crops.create_index("name", unique=True)
    await db.diseases.create_index("slug", unique=True)
    await db.predictions.create_index("user_id")


async def close_mongo_connection() -> None:
    global client, db
    if client is not None:
        client.close()
    client = None
    db = None


def get_database() -> AsyncIOMotorDatabase:
    if db is None:
        raise RuntimeError("MongoDB has not been initialized.")
    return db
