from fastapi import APIRouter

from app.api.endpoints.health import router as health_router
from app.domains.commit.router import router as commit_router
from app.domains.meeting_analysis.router import router as meeting_analysis_router
from app.domains.transcribe.router import router as transcribe_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(transcribe_router, prefix="/api")
api_router.include_router(meeting_analysis_router, prefix="/api")
api_router.include_router(commit_router, prefix="/api")
