from fastapi import APIRouter

from app.api.endpoints.health import router as health_router
from app.domains.decision.router import router as decision_router
from app.domains.transcribe.router import router as transcribe_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(transcribe_router, prefix="/api")
api_router.include_router(decision_router, prefix="/api")
