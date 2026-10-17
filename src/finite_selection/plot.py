"""Love plots of covariate imbalance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .inference import tasmd

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def asmd(x: pd.DataFrame, z: np.ndarray, groups: tuple[object, object]) -> pd.Series:
    """Absolute standardized mean difference between two groups.

    ``|mean_a - mean_b| / sqrt((sd_a^2 + sd_b^2) / 2)``.

    Args:
        x: Covariates.
        z: Group label per unit.
        groups: The two labels compared.

    Returns:
        ASMD per covariate.
    """
    df = pd.DataFrame(x)
    z = np.asarray(z)
    a, b = df[z == groups[0]], df[z == groups[1]]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()).abs() / pooled


def love_plot(
    x: pd.DataFrame,
    designs: dict[str, np.ndarray],
    *,
    imbalance: Literal["tasmd", "asmd"] = "tasmd",
    treat_lab: object = 1,
    mean_tar: np.ndarray | None = None,
    sd_tar: np.ndarray | None = None,
    denom: Literal["target", "group"] = "target",
    vline: float | None = None,
    xupper: float = 1.0,
    ax: Axes | None = None,
) -> Axes:
    """Plot imbalance per covariate for one or two assignments.

    Args:
        x: Covariates.
        designs: Named assignment vectors, one marker series each.
        imbalance: TASMD of group ``treat_lab``, or ASMD between the two
            smallest labels.
        treat_lab: Group assessed by TASMD.
        mean_tar: Target means for TASMD.
        sd_tar: Target standard deviations for TASMD.
        denom: Denominator for TASMD; see :func:`~finite_selection.tasmd`.
        vline: Draw a reference line at this imbalance.
        xupper: Upper limit of the imbalance axis.
        ax: Axes to draw on; a new figure is made if omitted.

    Returns:
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame(x)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.35 * df.shape[1] + 1.5))
    rows = np.arange(df.shape[1])
    # An open circle and a cross stay distinguishable when designs tie.
    markers = ["o", "x", "^", "D"]
    for i, (name, z) in enumerate(designs.items()):
        z = np.asarray(z)
        if imbalance == "asmd":
            labels = np.unique(z)
            values = asmd(df, z, (labels[0], labels[1]))
        else:
            values = tasmd(
                df,
                z,
                treat_lab=treat_lab,
                mean_tar=mean_tar,
                sd_tar=sd_tar,
                denom=denom,
            )
        marker = markers[i % len(markers)]
        color = f"C{i}"
        ax.scatter(
            values.to_numpy(),
            rows,
            marker=marker,
            label=name,
            zorder=3,
            color=color,
            facecolors="none" if marker != "x" else color,
        )
    ax.set_yticks(rows, df.columns)
    ax.invert_yaxis()
    ax.set_xlabel(imbalance.upper())
    ax.set_xlim(-0.02 * xupper, xupper)
    ax.axvline(0, linestyle="--", color="gray", linewidth=1)
    if vline is not None:
        ax.axvline(vline, linestyle="--", color="gray", linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False)
    return ax
