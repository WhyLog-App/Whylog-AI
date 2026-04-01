from dotenv import load_dotenv
from fastapi import FastAPI

from routers import transcribe

# .env 파일의 환경변수 로드 (DEEPGRAM_API_KEY 등)
load_dotenv()

app = FastAPI(title="WhyLog FastAPI", version="1.0.0")

# 라우터 등록 — 각 도메인별 엔드포인트를 여기서 연결
app.include_router(transcribe.router)


# 서버 동작 확인용
@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "FastAPI server is running"}


# 헬스체크 — 서버가 살아있는지 확인 (배포 환경에서 주로 사용)
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
