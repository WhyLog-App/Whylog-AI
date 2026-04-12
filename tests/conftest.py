import pytest

import app.core.chroma as chroma_module


@pytest.fixture(autouse=True)
def reset_chroma_client_singleton() -> None:
    """테스트 간 Chroma 클라이언트 싱글턴 오염을 방지한다."""
    chroma_module._client = None
    yield
    chroma_module._client = None
