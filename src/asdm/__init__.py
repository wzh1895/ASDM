from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("asdm")
except PackageNotFoundError:
    # Package is not installed, use a fallback
    __version__ = "unknown"

from .asdm import sdmodel, Parser, Solver
