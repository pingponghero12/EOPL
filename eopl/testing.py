from .pipeline import Pipeline

class MockPipelineFactory:
    """
    Constructs a minimal, fully initialized Pipeline object with synthetic data
    of configurable dimensions for unit testing operations.
    """
    @staticmethod
    def create(act: int = 100, alt: int = 100, channels: int = 10) -> Pipeline:
        pipe = Pipeline()
        pipe.set_run_options(use_cuda=False)
        # Generate dummy xarray dataset and assign to pipe.data
        return pipe
