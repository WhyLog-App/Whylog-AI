import os

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    app_name: str = Field(
        default_factory=lambda: os.getenv("APP_NAME", "WhyLog FastAPI")
    )
    app_version: str = Field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))


settings = AppSettings()
