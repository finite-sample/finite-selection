"""fsm and fsm_batch against the R package FSM 1.0.0."""

import numpy as np
import pytest
from conftest import fixture_names, load, r_data, r_matrix

from evenhand import fsm, fsm_batch, lalonde

# R names its options with spaces and capitals; the Python API uses snake case.
R_ARGS = {
    "s_function": "s_function",
    "Q_initial": "q_initial",
    "eps": "eps",
    "intercept": "intercept",
    "standardize": "standardize",
    "Pol_mat": "pol_mat",
    "w_pol": "w_pol",
}
# Criteria agree to the last of R's 7 rounded decimals; a value sitting on a
# rounding boundary can land one unit apart, and large criteria (tiny eps)
# carry ~1e-11 relative floating-point noise.
CRIT_ATOL = 1.01e-7
CRIT_RTOL = 1e-9


def assert_criteria_match(ours: np.ndarray, r_rows) -> None:
    theirs = r_matrix(r_rows)
    # R marks unavailable units with a -1 sentinel before turning it into NA,
    # so a genuine criterion of exactly -1 also comes back NA.
    sentinel = np.isnan(theirs) & (ours == -1)
    theirs[sentinel] = -1
    np.testing.assert_allclose(ours, theirs, rtol=CRIT_RTOL, atol=CRIT_ATOL)


DETERMINISTIC = [n for n in fixture_names("fsm_") if "ties_random" not in n]


def python_args(r_args: dict) -> dict:
    kw = {R_ARGS[k]: v for k, v in r_args.items()}
    kw["s_function"] = kw.get("s_function", "Dopt").replace(" ", "_").lower()
    for key in ("q_initial", "pol_mat"):
        if key in kw:
            kw[key] = np.array(kw[key], float)
    return kw


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_fsm_matches_r(name):
    case = load(name)
    x = r_data(case["dataset"]).to_numpy()
    kw = python_args(case["args"])
    if "error" in case:
        with pytest.raises(ValueError, match="collinear"):
            fsm(x, np.array(case["som"]), ties="smallest", **kw)
        return
    res = fsm(
        x,
        np.array(case["som"]),
        ties="smallest",
        keep_criteria="criteria" in case,
        **kw,
    )
    np.testing.assert_array_equal(res.assignment, case["z"])
    np.testing.assert_array_equal(res.selected + 1, case["selected"])
    if "criteria" in case:
        assert_criteria_match(res.criteria, case["criteria"])


def test_bundled_lalonde_has_declared_dtypes():
    df = lalonde()
    assert df.shape == (445, 8)
    assert (df.dtypes == np.float64).all()


def test_bundled_lalonde_matches_r():
    np.testing.assert_allclose(
        lalonde().to_numpy(), r_data("lalonde").to_numpy(), rtol=0, atol=1e-9
    )


def test_ties_random_distribution(rng):
    case = load("fsm_ties_random_binary")
    x = r_data("binary").to_numpy()
    reps = 2000
    z = np.array(
        [fsm(x, np.array(case["som"]), rng=rng).assignment for _ in range(reps)]
    )
    ours = (z == 1).mean(axis=0)
    theirs = np.array(case["freq_group1"])
    # Two independent Monte Carlo estimates of each unit's chance of group 1.
    se = np.sqrt(ours * (1 - ours) / reps + theirs * (1 - theirs) / case["reps"]).clip(
        1e-3
    )
    assert np.all(np.abs(ours - theirs) < 5 * se)


@pytest.mark.parametrize("name", fixture_names("batch_"))
def test_fsm_batch_matches_r(name):
    case = load(name)
    x = r_data(case["dataset"]).to_numpy()
    split = case["split"]
    res = fsm_batch(
        x[split:],
        x[:split],
        np.array(case["past_z"]),
        np.array(case["som"]),
        ties="smallest",
        intercept=case["intercept"],
        keep_criteria=True,
    )
    np.testing.assert_array_equal(res.assignment, case["z"])
    # R reports the original row index; Python counts within the new batch.
    np.testing.assert_array_equal(res.selected + split + 1, case["selected"])
    assert_criteria_match(res.criteria, case["criteria"])
