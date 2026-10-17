"""The Finite Selection Model allocation loop.

Groups take turns, in the order a :class:`~evenhand.som.SelectionOrder`
gives, and at each turn the choosing group takes the remaining unit that
scores highest on a selection function (Morris 1979; Chattopadhyay, Morris
and Zubizarreta 2021). Numerical conventions follow the R package ``FSM``
1.0.0 so that allocations agree unit for unit; ``docs/behavior.md`` lists
each convention and the reference fixture that pins it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import lapack

from .som import SelectionOrder

SelectionFunction = Literal[
    "constant",
    "dopt",
    "aopt",
    "negative_dopt",
    "max_pc",
    "min_pc",
    "dopt_pc",
    "max_average",
    "min_average",
    "dopt_average",
    "marginal_var_sum",
]
_MATRIX_FUNCTIONS = {"dopt", "aopt", "negative_dopt"}

# A group's information matrix counts as ill-conditioned below this reciprocal
# condition number, and is then ridged with eps * Q0 / N.
_RCOND_FLOOR = 1e-6
# Criteria are rounded before the argmax so near-ties resolve by tie rule
# rather than by floating-point noise.
_CRITERION_DIGITS = 7


@dataclass(frozen=True)
class FSMResult:
    """Outcome of an FSM allocation.

    Attributes:
        assignment: Group label of each unit, in input order.
        selected: Unit chosen at each stage (0-based row positions).
        criteria: Stage-by-unit matrix of selection-function values, NaN for
            units no longer available; ``None`` unless requested.
    """

    assignment: np.ndarray
    selected: np.ndarray
    criteria: np.ndarray | None = None


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)


def _rcond_1norm(m: np.ndarray) -> float:
    """LAPACK's 1-norm reciprocal condition estimate of the R factor of ``m``.

    The estimate, not the exact value, is what the regularization switch is
    defined on, so near the threshold the two can pick different branches.
    """
    r = np.linalg.qr(m, mode="r")
    rcond, info = lapack.dtrcon(  # pyright: ignore[reportAttributeAccessIssue]
        r, norm="1", uplo="U", diag="N"
    )
    return float(rcond) if info == 0 else 0.0


def _group_inverse(
    gram: np.ndarray, count: int, stage: int, ridge: np.ndarray
) -> np.ndarray:
    if count == 0:
        return np.linalg.inv(ridge)
    if _rcond_1norm(gram / stage) < _RCOND_FLOOR:
        return np.linalg.inv(gram / count + ridge)
    return np.linalg.inv(gram)


def _spread(values: np.ndarray, members: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Sum of squared deviations of ``members`` plus each candidate."""
    base = values[members]
    total = base.sum()
    total_sq = (base**2).sum()
    m = len(base) + 1
    cand = values[avail]
    return total_sq + cand**2 - (total + cand) ** 2 / m


def _marginal_var_sum(
    x: np.ndarray, members: np.ndarray, avail: np.ndarray
) -> np.ndarray:
    if x.shape[1] == 1:
        # R reports the sum of squares, not the variance, for one covariate.
        # Every candidate at a stage shares the group size, so the choice is
        # the same either way; only the reported criterion differs.
        return _spread(x[:, 0], members, avail)
    if len(members) == 0:
        return np.zeros(len(avail))
    return np.array(
        [
            np.trace(np.atleast_2d(np.cov(x[np.append(members, u)], rowvar=False)))
            for u in avail
        ]
    )


def _first_pc(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[0]


def fsm(
    x: np.ndarray,
    order: SelectionOrder | np.ndarray,
    *,
    s_function: SelectionFunction = "dopt",
    q_initial: np.ndarray | None = None,
    eps: float = 1e-3,
    ties: Literal["random", "smallest"] = "random",
    intercept: bool = True,
    standardize: bool = True,
    pol_mat: np.ndarray | None = None,
    w_pol: np.ndarray | None = None,
    keep_criteria: bool = False,
    rng: np.random.Generator | None = None,
) -> FSMResult:
    """Allocate units to groups with the Finite Selection Model.

    Args:
        x: Covariates, one row per unit.
        order: Selection order; ``order[i]`` is the group choosing at stage i.
            Must have one stage per unit.
        s_function: Selection function. ``"dopt"`` (D-optimality) picks the
            unit that most increases the determinant of the choosing group's
            information matrix.
        q_initial: Matrix used in place of the full-sample X'X to ridge an
            ill-conditioned group information matrix.
        eps: Ridge scale; the ridge is ``eps * Q0 / N``.
        ties: Pick a random maximizer, or the earliest-listed one.
        intercept: Include an intercept column in the design matrix.
        standardize: Standardize covariates (sample SD) before selection.
        pol_mat: Policy matrix for ``"aopt"``; one column per design column.
        w_pol: Policy weights for ``"aopt"``.
        keep_criteria: Also return the stage-by-unit criterion matrix.
        rng: Random generator for tie-breaking.

    Returns:
        The allocation.

    Raises:
        ValueError: If the order does not have one stage per unit, or the
            covariates are collinear and no ``q_initial`` is given.
    """
    x_cov = np.asarray(x, dtype=float)
    if x_cov.ndim == 1:
        x_cov = x_cov[:, None]
    labels = np.asarray(order.treat if isinstance(order, SelectionOrder) else order)
    n = len(x_cov)
    if len(labels) != n:
        raise ValueError("the selection order needs one stage per unit")
    return _allocate(
        x_cov,
        labels,
        past=None,
        s_function=s_function,
        q_initial=q_initial,
        eps=eps,
        ties=ties,
        intercept=intercept,
        standardize=standardize,
        pol_mat=pol_mat,
        w_pol=w_pol,
        keep_criteria=keep_criteria,
        rng=rng or np.random.default_rng(),
    )


def _allocate(
    x_cov: np.ndarray,
    labels: np.ndarray,
    *,
    past: tuple[np.ndarray, np.ndarray] | None,
    s_function: str,
    q_initial: np.ndarray | None,
    eps: float,
    ties: str,
    intercept: bool,
    standardize: bool,
    pol_mat: np.ndarray | None,
    w_pol: np.ndarray | None,
    keep_criteria: bool,
    rng: np.random.Generator,
) -> FSMResult:
    n = len(x_cov)
    x_past, z_past = past if past is not None else (x_cov[:0], np.array([]))
    n_past = len(x_past)
    # Past units shape the scaling, the ridge and each group's starting
    # information, but never become candidates.
    x_all = np.vstack([x_past, x_cov])
    if standardize:
        x_all = _standardize(x_all)
    design_all = np.column_stack([np.ones(len(x_all)), x_all]) if intercept else x_all
    if q_initial is None:
        q0 = design_all.T @ design_all
        if np.linalg.matrix_rank(q0) < q0.shape[0]:
            raise ValueError(
                "covariates are collinear, so X'X is singular; drop redundant "
                "columns or pass q_initial (e.g. an identity matrix)"
            )
    else:
        q0 = np.asarray(q_initial, float)
    ridge = eps * q0 / len(x_all)
    x_cov = x_all[n_past:]
    design = design_all[n_past:]
    design_past = design_all[:n_past]

    groups = list(dict.fromkeys(labels.tolist() + z_past.tolist()))
    gram = {g: design_past[z_past == g].T @ design_past[z_past == g] for g in groups}
    count = {g: int((z_past == g).sum()) for g in groups}
    members: dict[object, list[int]] = {g: [] for g in groups}

    t_mat = np.zeros((design.shape[1],) * 2)
    if s_function == "aopt":
        if pol_mat is None or w_pol is None:
            raise ValueError("aopt needs pol_mat and w_pol")
        pol = np.atleast_2d(np.asarray(pol_mat, float))
        t_mat = pol.T @ np.diag(np.atleast_1d(w_pol)) @ pol
    pc = (
        _first_pc(x_cov)
        if s_function in {"max_pc", "min_pc", "dopt_pc"}
        else np.empty(0)
    )
    row_mean = x_cov.mean(axis=1)

    z = np.full(n, -1, dtype=object)
    taken = np.zeros(n, dtype=bool)
    selected = np.empty(n, dtype=int)
    criteria = np.full((n, n), np.nan) if keep_criteria else None

    for stage, g in enumerate(labels.tolist()):
        avail = np.flatnonzero(~taken)
        current = np.asarray(members[g], dtype=int)
        if s_function in _MATRIX_FUNCTIONS:
            sn = _group_inverse(gram[g], count[g], stage + 1, ridge)
            xa = design[avail]
            quad = np.einsum("ij,jk,ik->i", xa, sn, xa)
            if s_function == "dopt":
                crit = quad
            elif s_function == "negative_dopt":
                crit = -quad
            else:
                num = np.einsum("ij,jk,ik->i", xa, sn @ t_mat @ sn, xa)
                crit = num / (1 + quad)
        elif s_function == "constant":
            crit = np.full(len(avail), 10.0)
        elif s_function == "max_pc":
            crit = pc[avail]
        elif s_function == "min_pc":
            crit = -pc[avail]
        elif s_function == "dopt_pc":
            crit = _spread(pc, current, avail)
        elif s_function == "max_average":
            crit = row_mean[avail]
        elif s_function == "min_average":
            crit = -row_mean[avail]
        elif s_function == "dopt_average":
            crit = _spread(row_mean, current, avail)
        elif s_function == "marginal_var_sum":
            crit = _marginal_var_sum(x_cov, current, avail)
        else:
            raise ValueError(f"unknown selection function {s_function!r}")

        crit = np.round(crit, _CRITERION_DIGITS)
        if criteria is not None:
            criteria[stage, avail] = crit
        best = avail[crit == crit.max()]
        pick = int(best[0]) if ties == "smallest" else int(rng.choice(best))
        z[pick] = g
        taken[pick] = True
        selected[stage] = pick
        members[g].append(pick)
        gram[g] += np.outer(design[pick], design[pick])
        count[g] += 1

    assignment = np.array(z.tolist())
    return FSMResult(assignment, selected, criteria)


def fsm_batch(
    x: np.ndarray,
    x_past: np.ndarray,
    z_past: np.ndarray,
    order: SelectionOrder | np.ndarray,
    *,
    s_function: SelectionFunction = "dopt",
    q_initial: np.ndarray | None = None,
    eps: float = 1e-3,
    ties: Literal["random", "smallest"] = "random",
    intercept: bool = True,
    standardize: bool = True,
    pol_mat: np.ndarray | None = None,
    w_pol: np.ndarray | None = None,
    keep_criteria: bool = False,
    rng: np.random.Generator | None = None,
) -> FSMResult:
    """Allocate a new batch of units given units already allocated.

    Each group starts from the information carried by its past members, and
    standardization and the ridge use past and new units together, so a
    sequence of batches behaves like one FSM run over a growing sample.

    Args:
        x: Covariates of the new units.
        x_past: Covariates of the units already allocated.
        z_past: Group labels of the units already allocated.
        order: Selection order for the new units, one stage per new unit.
        s_function: Selection function; see :func:`fsm`.
        q_initial: See :func:`fsm`.
        eps: See :func:`fsm`.
        ties: See :func:`fsm`.
        intercept: See :func:`fsm`.
        standardize: See :func:`fsm`.
        pol_mat: See :func:`fsm`.
        w_pol: See :func:`fsm`.
        keep_criteria: See :func:`fsm`.
        rng: See :func:`fsm`.

    Returns:
        The allocation of the new units; ``selected`` indexes rows of ``x``.

    Raises:
        ValueError: If the inputs do not line up.
    """
    x_new = np.atleast_2d(np.asarray(x, dtype=float).T).T
    past = np.atleast_2d(np.asarray(x_past, dtype=float).T).T
    z_old = np.asarray(z_past)
    labels = np.asarray(order.treat if isinstance(order, SelectionOrder) else order)
    if len(labels) != len(x_new):
        raise ValueError("the selection order needs one stage per new unit")
    if len(z_old) != len(past) or past.shape[1] != x_new.shape[1]:
        raise ValueError("x_past and z_past must describe the same past units")
    return _allocate(
        x_new,
        labels,
        past=(past, z_old),
        s_function=s_function,
        q_initial=q_initial,
        eps=eps,
        ties=ties,
        intercept=intercept,
        standardize=standardize,
        pol_mat=pol_mat,
        w_pol=w_pol,
        keep_criteria=keep_criteria,
        rng=rng or np.random.default_rng(),
    )
