from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from app.database import get_database
from app.dependencies import get_current_user
from app.schemas.all_schemas import PasswordChange, TokenResponse, UserLogin, UserProfileUpdate, UserRegister, UserResponse
from app.utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
AVATAR_DIR = Path(__file__).resolve().parents[2] / "uploads" / "avatars"
AVATAR_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def serialize_user(user: dict) -> UserResponse:
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        role=user["role"],
        avatar_url=user.get("avatar_url"),
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


@router.patch("/profile", response_model=UserResponse)
async def update_profile(payload: UserProfileUpdate, current_user: dict = Depends(get_current_user)):
    database = get_database()
    email = str(payload.email).lower()
    name = payload.name.strip()
    if len(name) < 2:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Name must contain at least 2 characters")
    existing = await database.users.find_one(
        {"email": email, "_id": {"$ne": current_user["_id"]}}
    )
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already in use")

    updated = {"name": name, "email": email}
    await database.users.update_one({"_id": current_user["_id"]}, {"$set": updated})
    current_user.update(updated)
    return serialize_user(current_user)


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(payload: PasswordChange, current_user: dict = Depends(get_current_user)):
    if not verify_password(payload.current_password, current_user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")
    if verify_password(payload.new_password, current_user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Choose a password you have not used")

    database = get_database()
    await database.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"password_hash": hash_password(payload.new_password)}},
    )


@router.post("/avatar", response_model=UserResponse)
async def update_avatar(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    suffix = AVATAR_TYPES.get(file.content_type or "")
    if not suffix:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a JPEG, PNG or WebP image")
    content = await file.read()
    if not content or len(content) > 4 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile image must be smaller than 4 MB")

    try:
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError

        with Image.open(BytesIO(content)) as image:
            expected_format = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[file.content_type]
            if image.format != expected_format:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image content does not match its file type")
            if max(image.size) > 4096:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile image dimensions must be 4096 pixels or smaller")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is not a valid image") from exc

    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4().hex}{suffix}"
    (AVATAR_DIR / filename).write_bytes(content)
    avatar_url = f"/uploads/avatars/{filename}"
    database = get_database()
    await database.users.update_one({"_id": current_user["_id"]}, {"$set": {"avatar_url": avatar_url}})

    previous_url = current_user.get("avatar_url", "")
    if previous_url.startswith("/uploads/avatars/"):
        previous_path = (AVATAR_DIR / Path(previous_url).name).resolve()
        if previous_path.parent == AVATAR_DIR.resolve() and previous_path.exists():
            previous_path.unlink()
    current_user["avatar_url"] = avatar_url
    return serialize_user(current_user)
