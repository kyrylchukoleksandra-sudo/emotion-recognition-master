"""Service health check route."""

from fastapi import APIRouter
from backend.app.schemas.prediction import HealthCheckResponse

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse(
        status="ok",
        service="Emotion Recognition API",
        version="1.0.0",
    )