from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_db: str = Field(default="crop_disease_analyzer")
    jwt_secret: str = Field(default="change-this-to-a-long-random-secret")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=10080)
    frontend_origin: str = Field(default="http://localhost:5173")


settings = Settings()
