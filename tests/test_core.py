"""Allocation contracts checked independently of the R reference fixtures."""

import numpy as np
import pytest

from finite_selection import fsm, fsm_batch


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("s_function", ["dopt_average", "dopt_pc", "marginal_var_sum"])
def test_batch_spread_includes_one_prior_member(s_function, standardize):
    result = fsm_batch(
        np.array([20, 0, 10]),
        np.array([100]),
        np.array([1]),
        np.array([1, 2, 1]),
        s_function=s_function,
        standardize=standardize,
        ties="smallest",
        keep_criteria=True,
    )
    expected = np.array([3200, 5000, 4050], dtype=float)
    if standardize:
        expected /= np.var([100, 20, 0, 10], ddof=1)
    np.testing.assert_allclose(result.criteria[0], np.round(expected, 7))
    assert result.selected[0] == 1


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("s_function", ["dopt_average", "dopt_pc", "marginal_var_sum"])
def test_batch_spread_matches_direct_group_variation(s_function, standardize):
    past = np.array([[40, -30], [12, 18], [-6, 22]], dtype=float)
    x = np.array([[2, 8], [-3, 1], [7, -5], [10, 4], [-8, -7], [4, 12]])
    old_labels = np.array([10, 10, 30])
    order = np.array([10, 30, 50, 10, 50, 30])
    result = fsm_batch(
        x,
        past,
        old_labels,
        order,
        s_function=s_function,
        standardize=standardize,
        ties="smallest",
        keep_criteria=True,
    )
    combined = np.vstack([past, x])
    if standardize:
        combined = (combined - combined.mean(axis=0)) / combined.std(axis=0, ddof=1)
    if s_function == "dopt_pc":
        # Eigenvectors of the covariance independently define the PCA direction.
        _, vectors = np.linalg.eigh(np.cov(combined, rowvar=False))
        values = (combined - combined.mean(axis=0)) @ vectors[:, -1]
    else:
        values = combined.mean(axis=1)

    assigned = old_labels.tolist()
    rows = list(range(len(past)))
    available = list(range(len(x)))
    for stage, label in enumerate(order):
        group = [row for row, g in zip(rows, assigned, strict=True) if g == label]
        expected = []
        for candidate in available:
            members = [*group, candidate + len(past)]
            if s_function == "marginal_var_sum":
                score = combined[members].var(axis=0, ddof=1).sum() if group else 0
            else:
                scores = values[members]
                score = ((scores - scores.mean()) ** 2).sum()
            expected.append(round(float(score), 7))
        np.testing.assert_allclose(
            result.criteria[stage, available], expected, atol=1e-7
        )
        pick = available[int(np.argmax(expected))]
        assert result.selected[stage] == pick
        assert result.assignment[pick] == label
        available.remove(pick)
        assert np.isnan(result.criteria[stage + 1 :, pick]).all()
        assigned.append(label)
        rows.append(pick + len(past))
    np.testing.assert_array_equal(np.sort(result.selected), np.arange(len(x)))


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize(
    "s_function",
    ["dopt_average", "dopt_pc", "marginal_var_sum", "max_pc", "min_pc"],
)
def test_empty_history_matches_fresh_allocation(s_function, standardize):
    x = np.array([[2, 8], [-3, 1], [7, -5], [10, 4], [-8, -7], [4, 12]])
    order = np.array([1, 2, 3, 1, 3, 2])
    kwargs = {
        "s_function": s_function,
        "standardize": standardize,
        "ties": "smallest",
        "keep_criteria": True,
    }
    fresh = fsm(x, order, **kwargs)
    batch = fsm_batch(x, np.empty((0, 2)), np.array([]), order, **kwargs)
    np.testing.assert_array_equal(batch.assignment, fresh.assignment)
    np.testing.assert_array_equal(batch.selected, fresh.selected)
    np.testing.assert_array_equal(batch.criteria, fresh.criteria)


@pytest.mark.parametrize("s_function", ["max_pc", "min_pc"])
def test_pc_extrema_keep_incoming_batch_projection(s_function):
    x = np.array([[-3, 0], [-1, 0.1], [1, -0.1], [3, 0]])
    past = np.array([[0, 1000], [1, -1000]])
    order = np.array([1, 2, 1, 2])
    kwargs = {
        "s_function": s_function,
        "standardize": False,
        "ties": "smallest",
        "keep_criteria": True,
    }
    fresh = fsm(x, order, **kwargs)
    batch = fsm_batch(x, past, np.array([1, 2]), order, **kwargs)
    np.testing.assert_array_equal(batch.selected, fresh.selected)
    np.testing.assert_array_equal(batch.criteria, fresh.criteria)
