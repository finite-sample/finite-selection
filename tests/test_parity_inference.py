"""perm_test, ess_rand, ess_model, tasmd_rand and make_sq_inter against R.

Where R departs from its documented behaviour, Python implements the
documented behaviour and the test pins both: exact agreement on the inputs R
actually uses, and the departure itself.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import load, r_data

from evenhand import (
    TasmdSummary,
    ess_model,
    ess_rand,
    make_sq_inter,
    perm_test,
    tasmd_rand,
)

ALT = {"not equal": "two-sided", "greater": "greater", "less": "less"}
METHOD = {"marginal mean": "mean", "marginal rank": "rank"}
PERM = load("perm_test")


def run_perm(case, tau=None):
    return perm_test(
        np.array(case["y_obs"]),
        np.array(case["alloc"])[0],
        np.array(case["alloc"])[1:],
        contrast=case["contrast"],
        tau=case["tau"] if tau is None else tau,
        method=METHOD[case["method"]],
        alternative=ALT[case["alternative"]],
    )


@pytest.mark.parametrize(
    "case",
    [c for c in PERM if c["tau"] == 0],
    ids=lambda c: f"{c['id']}-{c['method']}-{c['alternative']}",
)
def test_perm_test_matches_r(case):
    res = run_perm(case)
    r = case["result"]["value"]
    assert res.stat_obs == pytest.approx(r["test_stat_obs"], abs=1e-12)
    np.testing.assert_allclose(res.stat_iter, r["test_stat_iter"][1:], atol=1e-12)
    # R compares in raw floating point, so a statistic that ties the observed
    # one exactly can be dropped by rounding noise. Python counts every tie;
    # R's p-value lies between the tie-free and the tie-inclusive value.
    stat, obs = np.asarray(r["test_stat_iter"]), r["test_stat_obs"]
    ties = np.isclose(np.abs(stat), abs(obs), rtol=0, atol=1e-9)
    assert r["p_value"] <= res.p_value + 1e-12
    assert r["p_value"] >= res.p_value - ties.mean() - 1e-12
    if not ties.any():
        assert res.p_value == pytest.approx(r["p_value"], abs=1e-12)


@pytest.mark.parametrize(
    "case",
    [c for c in PERM if c["tau"] != 0],
    ids=lambda c: f"{c['method']}-{c['alternative']}",
)
def test_perm_test_nonzero_tau_imputes_the_null(case):
    """R shifts the tau = 0 statistics by tau instead of imputing the null.

    Python imputes Y(1) = Y(2) + tau. Pin R's behaviour as the shift of the
    tau = 0 statistics, and check that Python's reference distribution
    genuinely depends on tau where R's does not.
    """
    r = case["result"]["value"]
    zero = run_perm(case, tau=0.0)
    np.testing.assert_allclose(
        np.array(r["test_stat_iter"])[1:], zero.stat_iter - case["tau"], atol=1e-12
    )
    if case["method"] == "marginal rank":
        with pytest.raises(ValueError, match="no-effect"):
            run_perm(case)
        return
    ours = run_perm(case)
    assert ours.stat_obs == pytest.approx(zero.stat_obs, abs=1e-12)
    assert not np.allclose(ours.stat_iter, zero.stat_iter)


def test_perm_test_imputation_by_hand():
    y = np.array([3.0, 5.0, 1.0, 2.0])
    z_obs = np.array([1, 1, 2, 2])
    alloc = np.array([[1, 1, 2, 2], [2, 2, 1, 1], [1, 2, 1, 2]])
    res = perm_test(y, z_obs, alloc, tau=2.0, exact=True)
    # Null: Y(1) = Y(2) + 2. Imputed Y(1) = [3, 5, 3, 4]; Y(2) = [1, 3, 1, 2].
    np.testing.assert_allclose(res.stat_iter, [4 - 1.5, 3.5 - 2, 3 - 2.5])
    # Every |stat - 2| is at least |2.5 - 2|.
    assert res.p_value == pytest.approx(1.0)
    # Under the null of no effect the outcomes stay fixed: stats 2.5, -2.5,
    # -1.5, of which two reach |2.5|.
    zero = perm_test(y, z_obs, alloc, exact=True)
    np.testing.assert_allclose(zero.stat_iter, [2.5, -2.5, -1.5])
    assert zero.p_value == pytest.approx(2 / 3)


def test_ess_rand_matches_r_on_the_iterations_r_reads():
    """R reads only the first N iterations (N = number of units)."""
    case = load("ess_rand")
    y = np.array(case["y_mat"])
    designs = [np.array(a) for a in case["assign"]]
    n = len(y)
    np.testing.assert_allclose(
        ess_rand([d[:n] for d in designs], y), case["result"], rtol=1e-12
    )
    np.testing.assert_allclose(
        ess_rand([d[:n] for d in designs], y, contrast=(-1, 1)),
        case["result_rev"],
        rtol=1e-12,
    )
    assert not np.allclose(ess_rand(designs, y), case["result"])


@pytest.mark.parametrize("name", ["ess_model", "ess_model_unequal"])
def test_ess_model_matches_r(name):
    case = load(name)
    # R takes one assignment per column; Python takes one per row.
    res = ess_model(
        np.array(case["x_cov"]), np.array(case["assign"]).T, np.array(case["y_mat"])
    )
    np.testing.assert_allclose(res, case["result"], rtol=1e-12)


TASMD = {c["id"]: c for c in load("tasmd_rand")["cases"]}


@pytest.mark.parametrize("cid", ["vec_target", "vec_group", "vec_lab2", "vec_custom"])
def test_tasmd_vectors_match_r(cid):
    case = TASMD[cid]
    a = case["args"]
    x = r_data("lalonde60")
    res = tasmd_rand(
        x,
        {"CRD": np.array(a["alloc1"]), "FSM": np.array(a["alloc2"])},
        treat_lab=a.get("treat_lab", 1),
        mean_tar=a.get("mean_tar"),
        sd_tar=a.get("sd_tar"),
        denom=a.get("denom", "target"),
    )
    assert isinstance(res, pd.DataFrame)
    r = case["result"]["value"]
    # R rounds to the requested 10 decimals.
    np.testing.assert_allclose(res.to_numpy(), r["values"], atol=5.1e-11)
    assert list(res.index) == r["rownames"]


def test_tasmd_matrices_match_r():
    case = TASMD["mat_target"]
    a = case["args"]
    res = tasmd_rand(
        r_data("lalonde60"),
        {"CRD": np.array(a["alloc1"]), "FSM": np.array(a["alloc2"])},
    )
    assert isinstance(res, TasmdSummary)
    r = case["result"]["value"]
    np.testing.assert_allclose(
        res.mean.to_numpy(), r["tasmd_mean"]["values"], atol=5.1e-11
    )
    np.testing.assert_allclose(res.sd.to_numpy(), r["tasmd_sd"]["values"], atol=5.1e-11)


def test_tasmd_designs_of_different_lengths():
    """R fails with 'subscript out of bounds'; nothing requires equal counts."""
    a = TASMD["mat_mismatch"]["args"]
    res = tasmd_rand(
        r_data("lalonde60"),
        {"CRD": np.array(a["alloc1"]), "FSM": np.array(a["alloc2"])},
    )
    assert isinstance(res, TasmdSummary)
    assert res.mean.shape == (8, 2)


MSI = load("make_sq_inter")


@pytest.mark.parametrize(
    "case",
    MSI["cases"],
    ids=lambda c: f"sq{c['is_square']}-int{c['is_inter']}-marg{c['keep_marginal']}",
)
def test_make_sq_inter_matches_r(case):
    data = pd.DataFrame(MSI["data"]["values"], columns=MSI["data"]["columns"])
    kw = {k: case[k] for k in ("is_square", "is_inter", "keep_marginal")}
    if "error" in case["result"]:
        with pytest.raises(ValueError, match="nothing to return"):
            make_sq_inter(data, **kw)
        return
    res = make_sq_inter(data, **kw)
    assert list(res.columns) == case["result"]["value"]["columns"]
    np.testing.assert_array_equal(res.to_numpy(), case["result"]["value"]["values"])


def test_love_plot_draws_the_tasmd_values():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from evenhand import love_plot, tasmd

    case = TASMD["vec_target"]
    x = r_data("lalonde60")
    designs = {
        "CRD": np.array(case["args"]["alloc1"]),
        "FSM": np.array(case["args"]["alloc2"]),
    }
    ax = love_plot(x, designs)
    for coll, (_, z) in zip(ax.collections, designs.items(), strict=True):
        np.testing.assert_allclose(coll.get_offsets()[:, 0], tasmd(x, z).to_numpy())
    assert [t.get_text() for t in ax.get_yticklabels()] == list(x.columns)
