"""Exceptions for feature extraction module."""


class FeatureExtractionError(Exception):
    """Base exception for feature extraction errors."""


class InvalidLandmarksError(FeatureExtractionError):
    """Raised when input landmarks array is invalid or incomplete."""


class ZeroDistanceError(FeatureExtractionError):
    """Raised when interocular or base distance is zero, causing division by zero."""