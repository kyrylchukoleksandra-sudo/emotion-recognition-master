"""API routes package."""

from backend.app.api.routes.features import router as features_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.predictions import router as predictions_router

__all__ = ["features_router", "health_router", "predictions_router"]