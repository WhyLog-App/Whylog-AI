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
    gemini_llm_model: str = Field(
        default_factory=lambda: os.getenv(
            "GEMINI_LLM_MODEL", "gemini-3.1-flash-lite-preview"
        )
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    )

    # Commit Embedding
    commit_collection: str = Field(
        default_factory=lambda: os.getenv("COMMIT_COLLECTION", "commit_embeddings")
    )

    # Embedding
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )
    application_collection: str = Field(
        default_factory=lambda: os.getenv(
            "APPLICATION_COLLECTION", "application_embeddings"
        )
    )

    # Logging
    log_dir: str = Field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))


settings = AppSettings()
