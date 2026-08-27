from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from app.database import get_database
from app.dependencies import get_current_user
from app.schemas.all_schemas import TokenResponse, UserLogin, UserRegister, UserResponse
from app.utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


def serialize_user(user: dict) -> UserResponse:
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        role=user["role"],
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserRegister):
    database = get_database()
    existing = await database.users.find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered")
    document = {
        "name": payload.name.strip(),
        "email": payload.email.lower(),
        "password_hash": hash_password(payload.password),
        "role": "farmer",
        "created_at": datetime.now(timezone.utc),
    }
    result = await database.users.insert_one(document)
    document["_id"] = result.inserted_id
    user = serialize_user(document)
    token = create_access_token(user.id, user.role)
    return TokenResponse(access_token=token, user=user)


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin):
    database = get_database()
    user = await database.users.find_one({"email": payload.email.lower()})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    profile = serialize_user(user)
    token = create_access_token(profile.id, profile.role)
    return TokenResponse(access_token=token, user=profile)


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    return serialize_user(current_user)
