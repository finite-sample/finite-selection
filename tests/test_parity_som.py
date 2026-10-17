"""som and crd against the R package FSM 1.0.0.

R and numpy use different random number generators, so draws cannot match
one for one. What is checked exactly is everything deterministic given a
draw (SCOMARS's conditional probabilities along R's realized path); the
draws themselves are compared as distributions.
"""

import numpy as np
import pytest
from conftest import fixture_names, load

from evenhand import crd, scomars_probs, som

R_METHODS = {
    "SCOMARS": "scomars",
    "global percentage": "global_percentage",
    "randomized chunk": "randomized_chunk",
}
REPS = 10000


def draw_som(cfg: dict, case: dict, rng) -> np.ndarray:
    return som(
        cfg["sizes"],
        method=R_METHODS[cfg["method"]],
        n_units=cfg.get("discard_n"),
        control=case["control"],
        rng=rng,
    ).treat


def close_frequencies(ours, theirs, n_ours, n_theirs):
    se = np.sqrt(ours * (1 - ours) / n_ours + theirs * (1 - theirs) / n_theirs)
    return np.all(np.abs(ours - theirs) <= 5 * se.clip(1e-3))


@pytest.mark.parametrize("name", fixture_names("som_scomars"))
def test_scomars_probabilities_on_r_path(name):
    case = load(name)
    one = case["one"]
    lowest = 0 if case["control"] else 1
    second = np.array(one["Treat"]) - lowest
    probs = scomars_probs(second, np.array(case["marginal_treat"]))
    # R prints these rounded to four decimals.
    np.testing.assert_allclose(np.round(probs, 4), one["Probs"], atol=1e-12)


@pytest.mark.parametrize("name", fixture_names("som_gp"))
def test_global_percentage_path_follows_rule(name):
    case = load(name)
    sizes = np.array(case["config"]["sizes"], float)
    counts = np.zeros(len(sizes))
    for label in case["one"]["Treat"]:
        crit = (counts + 0.5) / sizes
        assert crit[label - 1] == crit.min()
        counts[label - 1] += 1


@pytest.mark.parametrize("name", fixture_names("som_"))
def test_som_stage_distribution(name, rng):
    case = load(name)
    cfg = case["config"]
    draws = np.array([draw_som(cfg, case, rng) for _ in range(REPS)])
    for j, level in enumerate(case["levels"]):
        theirs = np.array(case["stage_freq"])[:, j]
        ours = (draws == level).mean(axis=0)
        assert close_frequencies(ours, theirs, REPS, case["reps"]), level


@pytest.mark.parametrize("name", fixture_names("crd_"))
def test_crd_distribution(name, rng):
    case = load(name)
    draws = np.array(
        [crd(case["sizes"], control=case["control"], rng=rng) for _ in range(REPS)]
    )
    assert sorted(np.unique(draws)) == case["levels"]
    for j, level in enumerate(case["levels"]):
        np.testing.assert_array_equal((draws == level).sum(axis=1), case["sizes"][j])
        theirs = np.array(case["unit_freq"])[:, j]
        assert close_frequencies(
            (draws == level).mean(axis=0), theirs, REPS, case["reps"]
        )
