"""FastAPI application entry point."""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import features_router, health_router, predictions_router
# from backend.app.api.auth import router as auth_router
from backend.app.core.config import settings
from backend.app.db import models
from backend.app.db.session import Base, engine
from backend.app.ml.landmarks.errors import LandmarkExtractionError, NoFaceDetectedError

# 1. Initialize FastAPI application instance ONCE
app = FastAPI(
    title=getattr(settings, "PROJECT_NAME", "Facial Emotion Recognition API"),
    version="1.0.0",
    description="REST API for facial emotion recognition and pgvector similarity search",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 2. Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Create database tables on startup
Base.metadata.create_all(bind=engine)

# 4. Custom exception handlers for landmark extraction and face detection errors
@app.exception_handler(NoFaceDetectedError)
@app.exception_handler(LandmarkExtractionError)
async def face_detection_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"Face detection failed: {str(exc)}", "code": "NO_FACE_DETECTED"},
    )

# 5. Register API routers
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(predictions_router, prefix="/api/v1", tags=["Predictions & Uploads"])
app.include_router(features_router, prefix="/api/v1", tags=["Features & Search"])
# app.include_router(auth_router, prefix="/api/v1", tags=["Authentication"])


@app.get("/", tags=["Root"])
def root():
    """Root endpoint providing links to documentation and health status."""
    return {
        "message": "Emotion Recognition API is running.",
        "docs": "/docs",
        "health": "/api/v1/health",
    }