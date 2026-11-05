"""
API routes for image upload, preprocessing, landmarks extraction, batch/video processing, and emotion prediction.
Endpoints: /upload, /preprocess, /landmarks, /predict, /batch-images, /video
"""

import os
import shutil
import tempfile
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.ml.landmarks.detector import FaceLandmarkDetector
from backend.app.ml.landmarks.errors import (
    LandmarkExtractionError,
    NoFaceDetectedError,
)
from backend.app.ml.preprocessing.errors import PreprocessingError
from backend.app.ml.preprocessing.image import (
    PreprocessedFace,
    preprocess_face_crop,
)
from backend.app.schemas.prediction import EmotionPredictionResponse
from backend.app.services.prediction_service import PredictionOrchestratorService

router = APIRouter(prefix="", tags=["Predictions & Uploads"])


@router.post("/batch-images")
async def process_batch_images(
    files: List[UploadFile] = File(...),
    is_public: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Processes multiple uploaded images for batch emotion recognition and feature extraction."""
    results = []
    service = PredictionOrchestratorService(db_session=db)

    for file in files:
        try:
            contents = await file.read()
            response, _ = service.process_image_bytes(
                image_bytes=contents,
                image_name=file.filename or "upload.jpg",
                save_to_db=True,      # Зберігаємо запис
                is_public=is_public,  # 👈 Передаємо фактичний прапорець із UI
            )
            results.append(response.model_dump())
        except Exception as exc:
            db.rollback()
            results.append({"filename": file.filename, "error": str(exc)})

    return {
        "status": "success",
        "processed_count": len(files),
        "results": results,
    }


@router.post("/predict", response_model=EmotionPredictionResponse)
async def predict_emotion(
    file: UploadFile = File(...),
    is_public: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Performs emotion classification based on feature vector."""
    try:
        contents = await file.read()
        service = PredictionOrchestratorService(db_session=db)
        response, _ = service.process_image_bytes(
            image_bytes=contents,
            image_name=file.filename or "upload.jpg",
            save_to_db=True,
            is_public=is_public,  # 👈 Оновлено передачу прапорця
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


@router.post("/video")
async def process_video(
    file: UploadFile = File(...),
    mode: str = Form("multi"),  # 'multi' or 'single'
    is_public: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Processes uploaded video file for frame-by-frame emotion tracking and temporal analytics."""
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video format. Please upload MP4, MOV, or AVI.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_video_path = temp_file.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not open video file for processing.",
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(1, int(fps / 2))

        service = PredictionOrchestratorService(db_session=db)
        timeline = []
        emotions_count = {}
        frame_count = 0
        last_saved_emotion = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            success, encoded_image = cv2.imencode(".jpg", frame)
            if not success:
                continue

            try:
                current_timestamp = round(frame_count / fps, 2)
                should_save = is_public and (last_saved_emotion is None)

                response, _ = service.process_image_bytes(
                    image_bytes=encoded_image.tobytes(),
                    image_name=f"{file.filename}_t_{current_timestamp}s.jpg",
                    save_to_db=should_save,
                )

                dom_emotion = response.dominant_emotion
                if last_saved_emotion != dom_emotion:
                    last_saved_emotion = dom_emotion

                emotions_count[dom_emotion] = emotions_count.get(dom_emotion, 0) + 1

                # Додаємо повний набір метрик для кожного кадру
                timeline.append({
                    "timestamp_sec": current_timestamp,
                    "frame_number": frame_count,
                    "dominant_emotion": dom_emotion,
                    "confidence": round(response.confidence, 2),
                    "valence": response.valence,
                    "arousal": response.arousal,
                    "probabilities": response.probabilities,
                    "bounding_box": response.bounding_box,
                })
            except Exception:
                continue

        cap.release()

        if not timeline:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No faces detected in the provided video.",
            )

        dominant_emotion = max(emotions_count, key=emotions_count.get)

        return {
            "status": "success",
            "video_name": file.filename,
            "mode": mode,
            "is_public": is_public,
            "summary": {
                "total_frames_analyzed": len(timeline),
                "dominant_emotion": dominant_emotion,
                "emotions_distribution": emotions_count,
            },
            "timeline": timeline,
        }

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Accepts image from client, performs basic validation."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        )
    contents = await file.read()
    return {"filename": file.filename, "size_bytes": len(contents), "status": "validated"}


@router.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """Applies face detection, alignment, normalization, and CLAHE, returning the processed image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image encoding.",
            )

        preprocessed: PreprocessedFace = preprocess_face_crop(
            bgr_img,
            color_order="BGR",
        )

        out_img = preprocessed.image.astype(np.uint8)

        success, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode preprocessed image.",
            )

        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

    except (NoFaceDetectedError, LandmarkExtractionError, PreprocessingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessing failed: {str(exc)}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/landmarks")
async def extract_landmarks(file: UploadFile = File(...)):
    """Extracts and returns coordinates of facial landmarks."""
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