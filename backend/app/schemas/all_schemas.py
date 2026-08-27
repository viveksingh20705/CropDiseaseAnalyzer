from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TopPrediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    id: str
    crop: str
    disease: str
    class_label: str
    is_healthy: bool
    severity: str
    confidence: float
    top3: list[TopPrediction]
    symptoms: list[str]
    treatment: list[str]
    prevention: list[str]
    image_url: str
    created_at: datetime


class CropSchema(BaseModel):
    name: str
    scientific_name: Optional[str] = None
    description: str
    growing_season: Optional[str] = None


class DiseaseSchema(BaseModel):
    name: str
    crop: str
    slug: Optional[str] = None
    severity: str = "Moderate"
    symptoms: list[str]
    treatment: list[str]
    prevention: list[str]
    description: Optional[str] = None
