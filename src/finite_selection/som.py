"""Selection order matrices: the random, fair order in which groups choose."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import Literal

import numpy as np

Method = Literal["scomars", "global_percentage", "randomized_chunk"]


@dataclass(frozen=True)
class SelectionOrder:
    """The order in which treatment groups take turns choosing units.

    Attributes:
        treat: Group label of the chooser at each stage, one entry per stage.
        probs: For SCOMARS, the conditional probability at each stage that the
            second group chooses, given the path so far. ``None`` otherwise.
    """

    treat: np.ndarray
    probs: np.ndarray | None = None

    def __len__(self) -> int:
        """Return the number of stages."""
        return len(self.treat)


def _scomars_next(p_next: float, chosen: int, cum_prev: float) -> float:
    gap = chosen - cum_prev
    return float(np.clip((p_next - max(0.0, gap)) / (1 - abs(gap)), 0.0, 1.0))


def scomars_probs(treat: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """Conditional SCOMARS probabilities along a realized two-group path.

    SCOMARS (Morris 1979) draws stage i's chooser so that the running count of
    second-group choices never strays more than one from its expected count.
    The probability the second group chooses at stage i + 1 is
    ``(p[i+1] - max(0, S_i - F_i)) / (1 - |S_i - F_i|)``, clipped to [0, 1],
    where ``S_i`` counts second-group choices so far and ``F_i`` is the
    cumulative sum of the marginal probabilities ``p``.

    Args:
        treat: Realized path as 0/1 indicators of a second-group choice.
        marginal: Marginal probability that the second group chooses at each
            stage.

    Returns:
        The conditional probability used at each stage.
    """
    treat = np.asarray(treat, dtype=int)
    p = np.asarray(marginal, dtype=float)
    cum = np.cumsum(p)
    out = np.empty(len(p))
    out[0] = p[0]
    chosen = treat[0]
    for i in range(len(p) - 1):
        out[i + 1] = _scomars_next(p[i + 1], chosen, cum[i])
        chosen += treat[i + 1]
    return out


def som(
    treat_sizes: list[int] | tuple[int, ...],
    *,
    method: Method = "scomars",
    n_units: int | None = None,
    control: bool = False,
    marginal_treat: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> SelectionOrder:
    """Generate a selection order matrix (SOM).

    Args:
        treat_sizes: Size of each treatment group. With ``control=True`` the
            first entry is the control group.
        method: ``"scomars"`` (two groups only), ``"global_percentage"`` or
            ``"randomized_chunk"``.
        n_units: Total number of units. When it exceeds ``sum(treat_sizes)``
            the remainder forms an extra "discard" group that also takes turns,
            labeled after the treatment groups.
        control: Label groups 0, 1, ..., g-1 instead of 1, 2, ..., g.
        marginal_treat: SCOMARS only. Marginal probability that the second
            group chooses at each stage; defaults to its share of units.
        rng: Random generator.

    Returns:
        The selection order.

    Raises:
        ValueError: If the sizes or method are inconsistent.
    """
    rng = rng or np.random.default_rng()
    sizes = [int(s) for s in treat_sizes]
    if n_units is not None and n_units > sum(sizes):
        sizes.append(n_units - sum(sizes))
    elif n_units is not None and n_units < sum(sizes):
        raise ValueError("n_units is smaller than the total of treat_sizes")
    n = sum(sizes)
    offset = 0 if control else 1

    if method == "scomars":
        if len(sizes) != 2:
            raise ValueError("SCOMARS needs exactly two groups")
        p = (
            np.full(n, sizes[1] / n)
            if marginal_treat is None
            else np.asarray(marginal_treat, dtype=float)
        )
        if len(p) != n:
            raise ValueError("marginal_treat needs one probability per stage")
        second = np.empty(n, dtype=int)
        probs = np.empty(n)
        cum = np.cumsum(p)
        s = 0
        for i in range(n):
            prob = p[0] if i == 0 else _scomars_next(p[i], s, cum[i - 1])
            probs[i] = prob
            second[i] = rng.random() < prob
            s += second[i]
        return SelectionOrder(second + offset, probs)

    if method == "global_percentage":
        counts = np.zeros(len(sizes))
        target = np.asarray(sizes, dtype=float)
        treat = np.empty(n, dtype=int)
        for i in range(n):
            crit = (counts + 0.5) / target
            best = np.flatnonzero(crit == crit.min())
            pick = rng.choice(best)
            treat[i] = pick
            counts[pick] += 1
        return SelectionOrder(treat + offset)

    if method == "randomized_chunk":
        hcf = reduce(math.gcd, sizes)
        chunk = np.repeat(np.arange(len(sizes)), [s // hcf for s in sizes])
        treat = np.concatenate([rng.permutation(chunk) for _ in range(hcf)])
        return SelectionOrder(treat + offset)

    raise ValueError(f"unknown method {method!r}")
