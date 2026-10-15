class EOPLError(Exception):
    """Base exception for all EOPL errors."""
    pass

class ShapeError(EOPLError):
    """Raised when an operation's shape contract is violated by the current data."""
    pass
