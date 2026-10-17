"""Finite Selection Model: randomized, balanced allocation of units to groups."""

from importlib.metadata import PackageNotFoundError, version

from .core import FSMResult, fsm, fsm_batch
from .datasets import lalonde
from .features import make_sq_inter
from .inference import (
    PermTestResult,
    TasmdSummary,
    crd,
    ess_model,
    ess_rand,
    perm_test,
    tasmd,
    tasmd_rand,
)
from .plot import asmd, love_plot
from .som import SelectionOrder, scomars_probs, som

try:
    __version__ = version("finite-selection")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"

__all__ = [
    "FSMResult",
    "PermTestResult",
    "SelectionOrder",
    "TasmdSummary",
    "asmd",
    "crd",
    "ess_model",
    "ess_rand",
    "fsm",
    "fsm_batch",
    "lalonde",
    "love_plot",
    "make_sq_inter",
    "perm_test",
    "scomars_probs",
    "som",
    "tasmd",
    "tasmd_rand",
]
