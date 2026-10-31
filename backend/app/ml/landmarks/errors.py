"""Exceptions for face detection and landmark analysis (Module 3.7)."""


class LandmarkError(Exception):
    """Base exception for landmark and face detection errors."""


class NoFaceDetectedError(LandmarkError):
    """Raised when no face is detected in the input frame."""


class MultipleFacesDetectedError(LandmarkError):
    """Raised when more than one face is detected and strict mode is enabled."""


class LandmarkExtractionError(LandmarkError):
    """Raised when face is detected but landmark coordinates cannot be extracted."""