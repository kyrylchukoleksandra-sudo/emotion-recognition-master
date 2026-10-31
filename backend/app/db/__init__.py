"""Module 3.10: Feature Persistence and Vector Search with pgvector."""

from backend.app.db.models import FUSED_VECTOR_DIMENSION, FacialFeatureRecord
from backend.app.db.repositories.feature_repository import FeatureRepository
from backend.app.db.session import Base, SessionLocal, engine, get_db

__all__ = [
    "FUSED_VECTOR_DIMENSION",
    "Base",
    "FacialFeatureRecord",
    "FeatureRepository",
    "SessionLocal",
    "engine",
    "get_db",
]