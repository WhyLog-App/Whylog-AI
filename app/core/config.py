import os

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    app_name: str = Field(
        default_factory=lambda: os.getenv("APP_NAME", "WhyLog FastAPI")
    )
    app_version: str = Field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))

    # Gemini
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # ChromaDB
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    )

    # Decision Embedding
    decision_embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "DECISION_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )
    decision_collection: str = Field(
        default_factory=lambda: os.getenv("DECISION_COLLECTION", "decision_embeddings")
    )


settings = AppSettings()
