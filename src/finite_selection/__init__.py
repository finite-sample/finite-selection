"""Finite Selection Model: randomized, balanced allocation of units to groups."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("finite-selection")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"
