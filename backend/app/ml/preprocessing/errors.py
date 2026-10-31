class PreprocessingError(ValueError):
    """Base exception for image preprocessing errors."""


class InvalidImageError(PreprocessingError):
    """The image is unreadable or has an unsupported format."""


class InvalidImageShapeError(PreprocessingError):
    """The image array has an unsupported shape."""


class InvalidPixelRangeError(PreprocessingError):
    """The pixel data type or range is invalid or ambiguous."""


class ImageTooSmallError(PreprocessingError):
    """The image crop is too small for preprocessing."""


class ImageTooLargeError(PreprocessingError):
    """The image exceeds the configured maximum dimensions."""


class ImageProcessingBackendError(PreprocessingError):
    """A configured image-processing operation could not be executed."""