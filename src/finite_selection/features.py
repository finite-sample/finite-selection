"""Covariate transformations for balancing beyond the means."""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def make_sq_inter(
    data: pd.DataFrame,
    *,
    is_square: bool = True,
    is_inter: bool = True,
    keep_marginal: bool = True,
) -> pd.DataFrame:
    """Add squares and pairwise products of the columns of ``data``.

    Feeding these to :func:`~finite_selection.fsm` balances second moments
    and interactions, which balancing the raw columns does not guarantee.

    Args:
        data: Covariates.
        is_square: Add ``col^2`` for each column.
        is_inter: Add ``a.b`` for each pair of columns.
        keep_marginal: Keep the original columns first.

    Returns:
        Marginal columns, then interactions, then squares, each in input
        column order.

    Raises:
        ValueError: If every option is off, leaving nothing to return.
    """
    if not (is_square or is_inter or keep_marginal):
        raise ValueError("nothing to return: enable at least one option")
    parts = []
    if keep_marginal:
        parts.append(data)
    if is_inter:
        parts.append(
            pd.DataFrame(
                {
                    f"{a}.{b}": data[a] * data[b]
                    for a, b in combinations(data.columns, 2)
                },
                index=data.index,
            )
        )
    if is_square:
        parts.append(
            pd.DataFrame(
                {f"{a}^2": data[a] ** 2 for a in data.columns}, index=data.index
            )
        )
    return pd.concat(parts, axis=1)
