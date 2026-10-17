# evenhand

[![PyPI](https://img.shields.io/pypi/v/evenhand)](https://pypi.org/project/evenhand/)
[![Downloads](https://api.pepy.tech/badge/evenhand)](https://pepy.tech/projects/evenhand)
[![CI](https://github.com/finite-sample/evenhand/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/evenhand/actions/workflows/ci.yml)
[![Docs](https://github.com/finite-sample/evenhand/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/evenhand/)

<!-- SPHINX-START -->

evenhand assigns units to treatment groups at random while keeping the
groups alike on their covariates. It implements the Finite Selection Model
(FSM) that Carl Morris built for the RAND Health Insurance Experiment, in
Python, with allocation results checked against the R package
[FSM](https://cran.r-project.org/package=FSM). It works with NumPy arrays and
pandas data frames and includes balance diagnostics and randomization tests.
The contribution is a tested Python implementation, documented corrections,
and reproducible examples; the method is due to the authors cited below.

## How it works

The groups take turns choosing units, like captains picking teams. A random
draw fixes the order of turns, under a rule that never lets one group get far
ahead of its share. On its turn a group takes the remaining unit that most
improves its own design: by default, the unit that most increases the
determinant of its X'X matrix (D-optimality). Every group chases the same goal,
so the groups come out balanced. The turn order is random, so
randomization-based inference still applies.

## Installation

Requires Python 3.12 or newer.

```bash
pip install evenhand
pip install "evenhand[data,plot]"   # example data and love plots
```

## Example

This splits the 445 men of the National Supported Work data into groups of
222 and 223, once with the FSM and once by complete randomization. The table
reports TASMD for the first group: the distance between its covariate means and
the full-sample means, in standard deviations.

```python
import numpy as np
import evenhand as eh

x = eh.lalonde()
rng = np.random.default_rng(2026)

order = eh.som([222, 223], method="scomars", rng=rng)  # who chooses when
fsm_z = eh.fsm(x, order, rng=rng).assignment  # who goes where
crd_z = eh.crd([222, 223], rng=rng)  # complete randomization

print(eh.tasmd_rand(x, {"CRD": crd_z, "FSM": fsm_z}).round(3))
```

```text
             CRD    FSM
Age        0.066  0.002
Education  0.004  0.004
Black      0.023  0.011
Hispanic   0.055  0.007
Married    0.043  0.007
Nodegree   0.018  0.004
Re74       0.016  0.006
Re75       0.006  0.002
```

The FSM group sits within 0.011 standard deviations of the full sample on
every covariate. The randomized group is off by up to 0.066, on age. That is
one draw of each. Over 200 draws, the average TASMD per covariate is 0.034 to
0.041 under complete randomization and 0.004 to 0.008 under the FSM.
`tasmd_rand` reports these averages when given a matrix of assignments.

The package also allocates units that arrive in batches (`fsm_batch`), offers
ten other selection functions, and covers the analysis side: Fisher
randomization tests (`perm_test`), effective sample sizes for comparing designs
(`ess_rand`, `ess_model`), squares and interactions of the covariates
(`make_sq_inter`), and love plots (`love_plot`). The
[user guide](https://finite-sample.github.io/evenhand/guide.html) runs a full experiment with them.

## What it assumes

The FSM needs every unit and its covariates before assignment starts, so it
suits experiments that enroll a known list, not ones that recruit as they go.
It balances the columns you give it. Balance on squares or interactions
requires passing them in, through `make_sq_inter` for example. Against outcome
components it cannot see, it can increase variance relative to complete
randomization. Covariate
balance alone does not guarantee more precise estimates for every outcome.
The [JRSS-A discussion](https://doi.org/10.1093/jrsssa/qnag030) examines this
robustness concern. Run time grows with the square of the sample: about 0.1 seconds for 445
units and 7 seconds for 5,000.

## Agreement with R

The script `tests/fixtures/make_r_fixtures.R` runs R's exported functions on a
grid of cases and saves their inputs and outputs. The test suite checks the
Python functions against those records:

- Given the same turn order, `fsm` reproduces R's allocation and selection
  sequence in all 67 cases, which span the 11 selection functions, two to four
  groups, and the options (intercept, scaling, ridge size, `Q_initial`,
  collinear data). `fsm_batch` does the same in 3 cases.
- Random draws (`som`, `crd`) match R in distribution over 10,000 draws. R and
  numpy use different random number generators, so individual draws differ.
- In several places R 1.0.0 departs from its own documentation, and Python
  follows the documentation. For example, R's `ess_rand` reads only the first N
  assignments, and its `perm_test` ignores a nonzero null effect. Fixes are
  proposed upstream, and [Behaviour and parity with R](https://finite-sample.github.io/evenhand/behavior.html) lists
  each departure with its test.

`perm_test` imputes the two-group sharp null for every supported contrast,
including when `tau=0`. Sampled reference assignments use the Monte Carlo
correction `(b + 1) / (B + 1)`; `exact=True` is for a complete, equally weighted
reference distribution. Rank tests and tests with more than two groups support
only the sharp no-effect null.

## Credits

Carl N. Morris introduced the FSM (1979, *Journal of Econometrics* 11, 43-61).
Ambarish Chattopadhyay, Carl N. Morris and José R. Zubizarreta recast it in the
potential-outcomes framework and wrote the R package FSM, described in
"Randomized and balanced allocation of units into treatment groups using the
finite selection model for R" (arXiv:2105.02393) and "Balanced and robust
randomized treatment assignments: the finite selection model for the Health
Insurance Experiment and beyond" (*JRSS-A* 189(4), 2026; arXiv:2205.09736).
This package is an independent implementation checked against theirs. The
bundled data is Dehejia and Wahba's subset of the National Supported Work
demonstration (LaLonde 1986), from
[NBER](https://users.nber.org/~rdehejia/nswdata.html).

Cite the papers when you use the method, and this package (`CITATION.cff`) when
you use the software.
