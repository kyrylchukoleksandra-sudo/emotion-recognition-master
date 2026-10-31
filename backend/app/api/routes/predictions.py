"""
API routes for image upload, preprocessing, landmarks extraction, and emotion prediction.
Endpoints: /upload, /preprocess, /landmarks, /predict
"""

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.ml.landmarks.detector import FaceLandmarkDetector
from backend.app.ml.landmarks.errors import (
    LandmarkExtractionError,
    NoFaceDetectedError,
)
from backend.app.schemas.prediction import EmotionPredictionResponse
from backend.app.services.prediction_service import PredictionOrchestratorService

router = APIRouter()


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """/upload: Accepts image from client, performs basic validation."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        )
    contents = await file.read()
    return {"filename": file.filename, "size_bytes": len(contents), "status": "validated"}


@router.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """/preprocess: Invokes preprocessing module for face detection and normalization."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image encoding.",
            )
        return {"filename": file.filename, "status": "preprocessed_successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/landmarks")
async def extract_landmarks(file: UploadFile = File(...)):
    """/landmarks: Extracts and returns coordinates of facial landmarks."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode image content.",
            )

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        with FaceLandmarkDetector() as detector:
            detection_res = detector.process_frame(rgb_img)

        return {
            "landmarks_count": len(detection_res.landmarks_normalized),
            "bounding_box": list(detection_res.bounding_box),
            "blendshapes_count": len(detection_res.blendshapes),
        }
    except (NoFaceDetectedError, LandmarkExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Face detection failed: {str(exc)}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/predict", response_model=EmotionPredictionResponse)
async def predict_emotion(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """/predict: Performs emotion classification based on feature vector."""
    try:
        contents = await file.read()
        service = PredictionOrchestratorService(db_session=db)
        response, _ = service.process_image_bytes(
            image_bytes=contents,
            image_name=file.filename or "upload.jpg",
            save_to_db=False,
        )
        return response
    except (NoFaceDetectedError, LandmarkExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Face detection failed: {str(exc)}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc