from pydantic import BaseModel, Field


class ChangedFile(BaseModel):
    file_name: str = Field(min_length=1, description="변경된 파일 경로")
    changed_code: str = Field(min_length=1, description="unified diff 형식의 변경 코드")


class CommitAnalyzeRequest(BaseModel):
    commit_id: int = Field(ge=0, description="커밋 ID")
    message: str = Field(min_length=1, description="커밋 메시지")
    changed_file_list: list[ChangedFile] = Field(
        min_length=1, description="변경된 파일 목록"
    )


class CommitAnalyzeResponse(BaseModel):
    commit_id: int = Field(description="커밋 ID")
    summary: str = Field(description="커밋 요약")
