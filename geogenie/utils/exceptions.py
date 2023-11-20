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
