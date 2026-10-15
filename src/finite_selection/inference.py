"""Design assessment and inference after allocation.

Effective sample size (ESS) compares designs by the precision of their effect
estimators: a design's ESS is ``N * Var_best / Var_design``, so the most
precise design in the collection scores N (Chattopadhyay, Morris and
Zubizarreta 2021, Section 2.4). TASMD measures how far a group's covariate
means sit from a target profile, in standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if TYPE_CHECKING:
    from collections.abc import Sequence


_TIE_RTOL = 1e-10


def crd(
    treat_sizes: Sequence[int],
    *,
    control: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw a completely randomized design.

    Args:
        treat_sizes: Size of each group; with ``control=True`` the first entry
            is the control group.
        control: Label groups 0, 1, ..., g-1 instead of 1, 2, ..., g.
        rng: Random generator.

    Returns:
        A group label per unit, uniformly over all assignments with these
        group sizes.
    """
    rng = rng or np.random.default_rng()
    offset = 0 if control else 1
    labels = np.repeat(np.arange(len(treat_sizes)), treat_sizes) + offset
    return rng.permutation(labels)


@dataclass(frozen=True)
class PermTestResult:
    """Result of a Fisher randomization test.

    Attributes:
        stat_obs: Test statistic under the observed assignment.
        stat_iter: Test statistic under each assignment in the reference set.
        p_value: Exact tail probability or adjusted Monte Carlo p-value.
    """

    stat_obs: float
    stat_iter: np.ndarray
    p_value: float


def _group_contrast(
    values: np.ndarray, z: np.ndarray, levels: np.ndarray, contrast: np.ndarray
) -> float:
    return float(
        sum(
            c * values[z == lev].mean() for c, lev in zip(contrast, levels, strict=True)
        )
    )


def perm_test(
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    alloc: np.ndarray,
    *,
    contrast: Sequence[float] = (1.0, -1.0),
    tau: float = 0.0,
    method: Literal["mean", "rank"] = "mean",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    exact: bool = False,
) -> PermTestResult:
    """Fisher randomization test of a sharp null hypothesis.

    For two groups the null is ``c_1 Y_i(1) + c_2 Y_i(2) = tau`` for every unit.
    Both coefficients must be nonzero. Under this null the
    missing potential outcomes are imputed from the observed ones, and the
    statistic ``sum_g c_g * mean_g`` (of outcomes, or of their ranks) is
    recomputed for each assignment in ``alloc``. Two-sided p-values measure
    distance from ``tau``. With more than two groups, only the sharp null of
    identical potential outcomes in every group is supported: ``tau=0`` and
    coefficients summing to zero. A single contrast constraint does not
    identify the missing potential outcomes in that case.

    Rank statistics are supported only under the no-effect null (``tau=0``
    and coefficients summing to zero). For a nonzero effect, outcome units
    and rank units cannot be used interchangeably.

    Args:
        y_obs: Observed outcome per unit.
        z_obs: Observed group label per unit.
        alloc: Reference assignments, one row per assignment. By default,
            supply independent draws from the design that produced ``z_obs``;
            do not prepend the observed assignment. Random draws that happen
            to match it are retained. With ``exact=True``, supply the full
            equally weighted assignment distribution, including ``z_obs``.
        contrast: Coefficient for each group, in sorted label order.
        tau: Value of the contrast under the sharp null.
        method: Use outcomes (``"mean"``) or their ranks (``"rank"``).
        alternative: Direction of the alternative hypothesis.
        exact: If True, return the fraction of reference statistics at least
            as extreme. Otherwise use ``(b + 1) / (B + 1)``, where ``b`` of
            ``B`` sampled statistics are at least as extreme, to include the
            observed statistic and avoid zero Monte Carlo p-values.

    Returns:
        The observed statistic, the reference statistics and the p-value.

    Raises:
        ValueError: If inputs are invalid or the null is unsupported.
    """
    y = np.asarray(y_obs, dtype=float)
    z0 = np.asarray(z_obs)
    a = np.asarray(alloc)
    if a.ndim == 1 and a.size:
        a = a[None, :]
    c = np.asarray(contrast, dtype=float)
    if y.ndim != 1 or z0.ndim != 1 or y.shape != z0.shape or not y.size:
        raise ValueError("y_obs and z_obs must be nonempty vectors of equal length")
    if not np.isfinite(y).all() or not np.isfinite(tau):
        raise ValueError("outcomes and tau must be finite")
    if a.ndim != 2 or not len(a) or a.shape[1] != len(y):
        raise ValueError("alloc must contain nonempty assignments of length len(y_obs)")
    if pd.isna(z0).any() or pd.isna(a).any():
        raise ValueError("assignment labels must not be missing")
    levels = np.unique(z0)
    if len(levels) < 2 or c.shape != levels.shape or not np.isfinite(c).all():
        raise ValueError(
            "contrast needs one finite coefficient per group, with >=2 groups"
        )
    if not np.isin(a, levels).all() or any(
        not np.any(a == lev, axis=1).all() for lev in levels
    ):
        raise ValueError("every reference assignment must contain all observed groups")
    if method not in ("mean", "rank") or alternative not in (
        "greater",
        "less",
        "two-sided",
    ):
        raise ValueError("unsupported method or alternative")
    no_effect = tau == 0 and c.sum() == 0
    if method == "rank" and not no_effect:
        raise ValueError("rank statistics require the no-effect null")
    if len(levels) == 2:
        if np.any(c == 0):
            raise ValueError("two-group nulls require two nonzero coefficients")
        y_a = np.where(z0 == levels[0], y, (tau - c[1] * y) / c[0])
        y_b = np.where(z0 == levels[1], y, (tau - c[0] * y) / c[1])
    else:
        if not no_effect or not np.any(c):
            raise ValueError("more than two groups require the no-effect null")
        y_a = y_b = y
    if exact and not np.any(np.all(a == z0, axis=1)):
        raise ValueError("an exact reference distribution must include z_obs")

    def statistic(z: np.ndarray) -> float:
        v = np.where(z == levels[0], y_a, y_b)
        v = rankdata(v) if method == "rank" else v
        return _group_contrast(v, z, levels, c)

    stat_obs = statistic(z0)
    stat_iter = np.array([statistic(z) for z in a])
    # Statistics such as rank means tie exactly in exact arithmetic but can
    # differ in the last bits; ties count as at least as extreme.
    tol = _TIE_RTOL * abs(stat_obs - tau if alternative == "two-sided" else stat_obs)
    if alternative == "greater":
        extreme = stat_iter >= stat_obs - tol
    elif alternative == "less":
        extreme = stat_iter <= stat_obs + tol
    else:
        extreme = np.abs(stat_iter - tau) >= abs(stat_obs - tau) - tol
    adjustment = 0 if exact else 1
    p = (np.count_nonzero(extreme) + adjustment) / (len(a) + adjustment)
    return PermTestResult(stat_obs, stat_iter, float(p))


def _observed_contrast(
    z: np.ndarray, y_mat: np.ndarray, levels: np.ndarray, contrast: np.ndarray
) -> float:
    return float(
        sum(
            c * y_mat[z == lev, j].mean()
            for j, (c, lev) in enumerate(zip(contrast, levels, strict=True))
        )
    )


def ess_rand(
    designs: Sequence[np.ndarray],
    y_mat: np.ndarray,
    *,
    contrast: Sequence[float] = (1.0, -1.0),
) -> np.ndarray:
    """Randomization-based effective sample size of each design.

    For each design, the difference-in-means estimator of the contrast is
    computed on every assignment, and its variance across assignments is the
    design's precision.

    Args:
        designs: One array per design, each with one assignment per row.
        y_mat: Potential outcomes, one column per group in sorted label order.
        contrast: Coefficient for each group.

    Returns:
        ESS per design; the most precise design has ESS equal to N.
    """
    y = np.asarray(y_mat, dtype=float)
    c = np.asarray(contrast, dtype=float)
    variances = []
    for d in designs:
        d = np.atleast_2d(np.asarray(d))
        levels = np.unique(d)
        est = np.array([_observed_contrast(z, y, levels, c) for z in d])
        variances.append(est.var(ddof=1))
    v = np.asarray(variances)
    return len(y) * v.min() / v


def ess_model(
    x_cov: np.ndarray,
    assignments: np.ndarray,
    y_mat: np.ndarray,
    *,
    contrast: Sequence[float] = (1.0, -1.0),
) -> np.ndarray:
    """Model-based effective sample size of each assignment.

    Within each group, the outcome is regressed on an intercept and
    ``x_cov``. The contrast is estimated by regression imputation at the
    full-sample covariate mean ``b``, with variance
    ``sum_g c_g^2 * s_g^2 * b' (B_g' B_g)^{-1} b``, where ``s_g^2`` is the
    group's residual variance on ``n_g - p`` degrees of freedom.

    Args:
        x_cov: Covariates used in the outcome models, one row per unit.
        assignments: Assignments to assess, one per row.
        y_mat: Potential outcomes, one column per group in sorted label order.
        contrast: Coefficient for each group.

    Returns:
        ESS per assignment; the most precise assignment has ESS equal to N.
    """
    x = np.asarray(x_cov, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    b = np.column_stack([np.ones(len(x)), x])
    b_bar = b.mean(axis=0)
    y = np.asarray(y_mat, dtype=float)
    c = np.asarray(contrast, dtype=float)
    a = np.atleast_2d(np.asarray(assignments))
    variances = []
    for z in a:
        levels = np.unique(z)
        v = 0.0
        for j, (cj, lev) in enumerate(zip(c, levels, strict=True)):
            bg = b[z == lev]
            yg = y[z == lev, j]
            beta, *_ = np.linalg.lstsq(bg, yg, rcond=None)
            resid = yg - bg @ beta
            s2 = resid @ resid / (len(yg) - b.shape[1])
            v += cj**2 * s2 * b_bar @ np.linalg.solve(bg.T @ bg, b_bar)
        variances.append(v)
    var = np.asarray(variances)
    return len(y) * var.min() / var


@dataclass(frozen=True)
class TasmdSummary:
    """TASMD across repeated assignments.

    Attributes:
        mean: Mean TASMD per covariate (rows) and design (columns).
        sd: Standard deviation of the TASMD across assignments.
    """

    mean: pd.DataFrame
    sd: pd.DataFrame


def tasmd(
    x: pd.DataFrame,
    z: np.ndarray,
    *,
    treat_lab: object = 1,
    mean_tar: np.ndarray | None = None,
    sd_tar: np.ndarray | None = None,
    denom: Literal["target", "group"] = "target",
) -> pd.Series:
    """Target absolute standardized mean difference of one group.

    ``|mean of the group - target| / sd``, where the target defaults to the
    full-sample mean and ``sd`` is the target SD (default: full-sample SD) or,
    with ``denom="group"``, the SD within the group.

    Args:
        x: Covariates, one column per covariate.
        z: Group label per unit.
        treat_lab: Label of the group assessed.
        mean_tar: Target covariate means.
        sd_tar: Target covariate standard deviations.
        denom: Which standard deviation to divide by.

    Returns:
        TASMD per covariate.
    """
    df = pd.DataFrame(x)
    z = np.asarray(z)
    grp = df[z == treat_lab]
    target = df.mean() if mean_tar is None else pd.Series(mean_tar, index=df.columns)
    if denom == "group":
        scale = grp.std(ddof=1)
    else:
        scale = (
            df.std(ddof=1) if sd_tar is None else pd.Series(sd_tar, index=df.columns)
        )
    return (grp.mean() - target).abs() / scale


def tasmd_rand(
    x: pd.DataFrame,
    designs: dict[str, np.ndarray],
    *,
    treat_lab: object = 1,
    mean_tar: np.ndarray | None = None,
    sd_tar: np.ndarray | None = None,
    denom: Literal["target", "group"] = "target",
) -> pd.DataFrame | TasmdSummary:
    """TASMD of one group under one or more designs.

    Args:
        x: Covariates, one column per covariate.
        designs: Named designs. Each value is one assignment (a vector) or a
            set of assignments (one per row); all values must be the same kind.
        treat_lab: Label of the group assessed.
        mean_tar: Target covariate means.
        sd_tar: Target covariate standard deviations.
        denom: Which standard deviation to divide by.

    Returns:
        For vectors, a covariate-by-design table of TASMDs. For sets of
        assignments, their mean and standard deviation across assignments.

    Raises:
        ValueError: If designs mix single assignments and sets.
    """
    df = pd.DataFrame(x)
    arrays = {k: np.asarray(v) for k, v in designs.items()}
    dims = {a.ndim for a in arrays.values()}
    if len(dims) != 1:
        raise ValueError("designs must be all vectors or all matrices")
    kw = {
        "treat_lab": treat_lab,
        "mean_tar": mean_tar,
        "sd_tar": sd_tar,
        "denom": denom,
    }
    if dims == {1}:
        return pd.DataFrame({k: tasmd(df, a, **kw) for k, a in arrays.items()})
    per = {k: pd.DataFrame([tasmd(df, z, **kw) for z in a]) for k, a in arrays.items()}
    return TasmdSummary(
        mean=pd.DataFrame({k: v.mean() for k, v in per.items()}),
        sd=pd.DataFrame({k: v.std(ddof=1) for k, v in per.items()}),
    )
