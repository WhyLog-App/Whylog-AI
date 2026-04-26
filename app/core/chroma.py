import chromadb

from app.core.config import settings

_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """ChromaDB PersistentClient 싱글턴을 반환한다."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_commit_collection() -> chromadb.Collection:
    """커밋 임베딩 컬렉션을 반환한다 (없으면 자동 생성)."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.commit_collection,
        metadata={"hnsw:space": "cosine"},
    )


def get_application_collection() -> chromadb.Collection:
    """적용사항 임베딩 컬렉션을 반환한다 (없으면 자동 생성)."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.application_collection,
        metadata={"hnsw:space": "cosine"},
    )
