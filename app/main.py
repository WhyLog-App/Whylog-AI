import logging
import logging.handlers
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.core.config import settings
from app.core.errors import AppServiceError
from app.core.logging_middleware import RequestLoggingMiddleware
from app.core.responses import error_response

# .env 파일의 환경변수 로드 (DEEPGRAM_API_KEY 등)
load_dotenv()

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "app.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(), _file_handler],
)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(api_router)
logger = logging.getLogger(__name__)


def _status_to_error_code(status_code: int) -> str:
    # HTTP 상태 코드를 서비스 공통 에러 코드로 매핑
    mapping = {
        400: "COMMON_400",
        401: "COMMON_401",
        403: "COMMON_403",
        404: "COMMON_404",
        413: "COMMON_413",
        422: "COMMON_422",
        429: "COMMON_429",
        500: "COMMON_500",
        502: "COMMON_502",
        504: "COMMON_504",
    }
    return mapping.get(status_code, f"COMMON_{status_code}")


@app.exception_handler(AppServiceError)
async def handle_app_service_error(
    request: Request, exc: AppServiceError
) -> JSONResponse:
    # 서비스 레이어 예외를 공통 에러 응답으로 변환
    _ = request
    code = _status_to_error_code(exc.status_code)
    payload = error_response(
        code=code,
        message=exc.message,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 요청 바디/파라미터 검증 실패 응답 처리
    _ = request
    payload = error_response(
        code="COMMON_422",
        message="요청 본문 또는 파라미터 검증에 실패했습니다.",
        result=exc.errors(),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    # 라우터에서 발생한 HTTP 예외를 공통 포맷으로 변환
    _ = request
    code = _status_to_error_code(exc.status_code)
    detail = exc.detail
    message = detail if isinstance(detail, str) else "요청 처리 중 오류가 발생했습니다."
    payload = error_response(
        code=code,
        message=message,
        result=None if isinstance(detail, str) else detail,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(StarletteHTTPException)
async def handle_starlette_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    # 미등록 경로(404) 등 Starlette 예외를 공통 포맷으로 변환
    _ = request
    code = _status_to_error_code(exc.status_code)
    detail = exc.detail
    message = detail if isinstance(detail, str) else "요청 처리 중 오류가 발생했습니다."
    payload = error_response(
        code=code,
        message=message,
        result=None if isinstance(detail, str) else detail,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    # 예상하지 못한 예외를 500 공통 응답으로 변환
    logger.exception("Unhandled exception", extra={"path": str(request.url.path)})
    payload = error_response(
        code="COMMON_500",
        message="서버 내부 오류가 발생했습니다.",
    )
    return JSONResponse(status_code=500, content=payload.model_dump())
