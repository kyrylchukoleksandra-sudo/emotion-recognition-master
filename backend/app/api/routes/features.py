"""
API routes for facial feature calculation, storage, and pgvector similarity search.
Endpoints: /features, /save_features, /search
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.db.repositories.feature_repository import FeatureRepository
from backend.app.db.session import get_db
from backend.app.ml.landmarks.errors import LandmarkExtractionError, NoFaceDetectedError
from backend.app.services.prediction_service import PredictionOrchestratorService

router = APIRouter()


@router.post("/features")
async def extract_features(file: UploadFile = File(...)):
    """/features: Calculates mimic/geometric features and builds fused feature vector."""
    try:
        contents = await file.read()
        service = PredictionOrchestratorService()
        _, fused_vector = service.process_image_bytes(
            image_bytes=contents, image_name=file.filename or "upload.jpg", save_to_db=False
        )
        return {
            "fused_vector_dim": len(fused_vector),
            "fused_vector": fused_vector.tolist(),
        }
    except (NoFaceDetectedError, LandmarkExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Face detection failed: {str(exc)}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/save_features")
async def save_features(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """/save_features: Persists feature vectors into PostgreSQL using pgvector extension."""
    try:
        contents = await file.read()
        service = PredictionOrchestratorService(db_session=db)
        response, fused_vector = service.process_image_bytes(
            image_bytes=contents, image_name=file.filename or "save_item.jpg", save_to_db=True
        )
        return {
            "status": "saved",
            "emotion_label": response.dominant_emotion,
            "vector_dim": len(fused_vector),
        }
    except (NoFaceDetectedError, LandmarkExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Face detection failed: {str(exc)}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/search")
async def search_similar_features(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
    filter_emotion: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """/search: Executes nearest-neighbor search using cosine similarity on feature vectors."""
    try:
        contents = await file.read()
        service = PredictionOrchestratorService(db_session=db)
        _, fused_vector = service.process_image_bytes(
            image_bytes=contents, image_name=file.filename or "search.jpg", save_to_db=False
        )

        repo = FeatureRepository(db)
        similar_records = repo.search_similar_features(
            query_vector=fused_vector, top_k=top_k, filter_emotion=filter_emotion
        )

        return {
            "top_k": top_k,
            "results": [
                {
                    "record_id": rec.id,
                    "image_name": rec.image_name,
                    "emotion_label": rec.emotion_label,
                    "similarity_score": score,
                }
                for rec, score in similar_records
            ],
        }
    except (NoFaceDetectedError, LandmarkExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Face detection failed: {str(exc)}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc