from typing import Any, Literal

from pydantic import BaseModel, Field


# 성공 응답 공통 래퍼 DTO
class ApiResponse[T](BaseModel):
    isSuccess: Literal[True] = Field(default=True, description="요청 성공 여부")
    code: str = Field(default="COMMON_200", description="응답 코드")
    message: str = Field(
        default="요청이 성공적으로 처리되었습니다.",
        description="응답 메시지",
    )
    result: T = Field(description="실제 응답 데이터")


# 실패 응답 공통 래퍼 DTO
class ApiErrorResponse(BaseModel):
    isSuccess: Literal[False] = Field(default=False, description="요청 성공 여부")
    code: str = Field(description="에러 코드")
    message: str = Field(description="에러 메시지")
    result: Any | None = Field(default=None, description="실패 시 기본적으로 null")


def ok_response[T](
    result: T,
    *,
    code: str = "COMMON_200",
    message: str = "요청이 성공적으로 처리되었습니다.",
) -> ApiResponse[T]:
    # 성공 응답을 공통 형태로 생성
    return ApiResponse[T](
        code=code,
        message=message,
        result=result,
    )


def error_response(
    *,
    code: str,
    message: str,
    result: Any | None = None,
) -> ApiErrorResponse:
    # 실패 응답을 공통 형태로 생성
    return ApiErrorResponse(
        code=code,
        message=message,
        result=result,
    )
