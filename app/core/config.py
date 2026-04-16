import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


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

    # Commit Embedding
    commit_collection: str = Field(
        default_factory=lambda: os.getenv("COMMIT_COLLECTION", "commit_embeddings")
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
