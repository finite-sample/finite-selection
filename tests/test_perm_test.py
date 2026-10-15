"""Randomization inference checked against small, explicit designs."""

from itertools import combinations

import numpy as np
import pytest

from finite_selection import perm_test


def balanced_assignments():
    rows = []
    for selected in combinations(range(4), 2):
        row = np.full(4, 2)
        row[list(selected)] = 1
        rows.append(row)
    return np.array(rows)


def test_zero_tau_still_imputes_general_two_group_null():
    result = perm_test(
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 2, 2],
        [[2, 2, 1, 1]],
        contrast=(1.0, 1.0),
        alternative="greater",
    )
    np.testing.assert_allclose(result.stat_iter, [-5.0])
    assert result.p_value == pytest.approx(0.5)


def test_exact_distribution_requires_observed_assignment():
    with pytest.raises(ValueError, match="include z_obs"):
        perm_test([1.0, 2.0], [1, 2], [[2, 1]], exact=True)


def test_no_effect_rank_test_preserves_ties_when_contrast_is_scaled():
    result = perm_test(
        [1.0, 3.0, 3.0, 3.0],
        [1, 1, 2, 2],
        balanced_assignments(),
        contrast=(0.1, -0.1),
        method="rank",
        alternative="less",
        exact=True,
    )
    np.testing.assert_allclose(result.stat_iter, [-0.1] * 3 + [0.1] * 3)
    assert result.p_value == pytest.approx(0.5)


def test_decimal_multigroup_contrast_under_no_effect():
    result = perm_test(
        [1.0, 2.0, 3.0],
        [1, 2, 3],
        [[1, 2, 3], [3, 2, 1]],
        contrast=(0.1, 0.2, -0.3),
        exact=True,
    )
    np.testing.assert_allclose(result.stat_iter, [-0.4, 0.4])
    assert result.p_value == 1.0


@pytest.mark.parametrize("tau", [0.0, 3.0])
@pytest.mark.parametrize("contrast", [(1.0, 1.0), (2.0, -1.0), (-1.0, 1.0)])
@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_general_null_by_enumeration(tau, contrast, alternative):
    y1 = np.array([1.0, 4.0, 2.0, 8.0])
    y2 = (tau - contrast[0] * y1) / contrast[1]
    assignments = balanced_assignments()
    observed = assignments[0]
    y_obs = np.where(observed == 1, y1, y2)
    expected = np.array(
        [
            contrast[0] * y1[z == 1].mean() + contrast[1] * y2[z == 2].mean()
            for z in assignments
        ]
    )
    if alternative == "greater":
        extreme = expected >= expected[0] - 1e-12
    elif alternative == "less":
        extreme = expected <= expected[0] + 1e-12
    else:
        extreme = np.abs(expected - tau) >= abs(expected[0] - tau) - 1e-12
    result = perm_test(
        y_obs,
        observed,
        assignments,
        contrast=contrast,
        tau=tau,
        alternative=alternative,
        exact=True,
    )
    np.testing.assert_allclose(result.stat_iter, expected, atol=1e-12)
    assert result.stat_obs == pytest.approx(expected[0])
    assert result.p_value == pytest.approx(extreme.mean())


def test_monte_carlo_pvalue_includes_observed_statistic():
    result = perm_test(
        [4.0, 5.0, 1.0, 2.0],
        [1, 1, 2, 2],
        [[2, 2, 1, 1], [1, 2, 1, 2]],
        alternative="greater",
    )
    assert result.p_value == pytest.approx(1 / 3)


@pytest.mark.parametrize("scale", [1.0, 1e-14, 1e14])
def test_pvalue_does_not_depend_on_outcome_units(scale):
    result = perm_test(
        scale * np.array([4.0, 5.0, 1.0, 2.0]),
        [1, 1, 2, 2],
        balanced_assignments(),
        alternative="greater",
        exact=True,
    )
    assert result.p_value == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contrast": (0.0, 1.0)},
        {"contrast": (float("nan"), -1.0)},
        {"tau": float("inf")},
        {"method": "other"},
        {"alternative": "other"},
        {"method": "rank", "tau": 1.0},
        {"method": "rank", "contrast": (1.0, 1.0)},
    ],
)
def test_rejects_unsupported_nulls_and_options(kwargs):
    with pytest.raises(ValueError, match=r"coefficient|finite|unsupported|no-effect"):
        perm_test([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2], balanced_assignments(), **kwargs)


@pytest.mark.parametrize("alloc", [[], [[1, 2]], [[1, 1, 1, 1]], [[1, 1, 2, 3]]])
def test_rejects_invalid_reference_assignments(alloc):
    with pytest.raises(ValueError, match=r"assignments?|groups"):
        perm_test([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2], alloc)


def test_multigroup_null_is_explicitly_no_effect():
    result = perm_test(
        [1.0, 2.0, 3.0],
        [1, 2, 3],
        [[1, 2, 3], [3, 2, 1]],
        contrast=(1.0, -1.0, 0.0),
        exact=True,
    )
    np.testing.assert_allclose(result.stat_iter, [-1.0, 1.0])
    assert result.p_value == 1.0
    with pytest.raises(ValueError, match="no-effect"):
        perm_test(
            [1.0, 2.0, 3.0],
            [1, 2, 3],
            [[1, 2, 3]],
            contrast=(1.0, 1.0, 1.0),
        )
