from fastapi import APIRouter

from app.core.responses import ApiResponse, ok_response

router = APIRouter(tags=["system"])


@router.get("/")
def read_root() -> ApiResponse[dict[str, str]]:
    return ok_response(
        {"message": "FastAPI server is running"},
        code="SYSTEM_200",
        message="서버 상태 조회에 성공했습니다.",
    )


@router.get("/health")
def health_check() -> ApiResponse[dict[str, str]]:
    return ok_response(
        {"status": "ok"},
        code="SYSTEM_200",
        message="헬스체크에 성공했습니다.",
    )
