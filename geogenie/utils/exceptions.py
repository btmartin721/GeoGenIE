class GPUUnavailableError(Exception):
    """Exception raised when a GPU is specified but not available."""

    def __init__(self, message="Specified GPU is not available."):
        self.message = message
        super().__init__(self.message)


class ResourceAllocationError(Exception):
    """Exception raised when a specified resource is invalid."""

    def __init__(self, message="Specified resource is not available."):
        self.message = message
        super().__init__(self.message)


class TimeoutException(Exception):
    pass


class DataStructureError(Exception):
    """Base class for exceptions in DataStructure."""

    pass


class InvalidSampleDataError(DataStructureError):
    """Exception raised for errors in the sample data file."""

    def __init__(
        self,
        message="Invalid sample data format. Expected a tab-delimited file with three columns: sampleID, x, and y.",
    ):
        self.message = message
        super().__init__(self.message)


class SampleOrderingError(DataStructureError):
    """Exception raised for errors in the sample ordering."""

    def __init__(self, message="Invalid sample ordering after filtering and sorting."):
        self.message = message
        super().__init__(self.message)


class InvalidInputShapeError(DataStructureError):
    """Exception raised for invalid input shapes."""

    def __init__(self, shape1, shape2):
        self.message = f"Invalid input shapes. The number of rows (samples) must be equal, but got: {shape1}, {shape2}"
        super().__init__(self.message)


class EmbeddingError(DataStructureError):
    """Exception raised for errors during embedding."""

    def __init__(self, message="n_components could not be estimated for embedding."):
        self.message = message
        super().__init__(self.message)


class OutlierDetectionError(DataStructureError):
    """Exception raised for errors during outlier detection."""

    def __init__(self, message="Error occurred during outlier detection."):
        self.message = message
        super().__init__(self.message)
