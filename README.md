# finite-selection

[![PyPI](https://img.shields.io/pypi/v/finite-selection)](https://pypi.org/project/finite-selection/)
[![CI](https://github.com/finite-sample/finite-selection/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/finite-selection/actions/workflows/ci.yml)
[![Docs](https://github.com/finite-sample/finite-selection/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/finite-selection/)
[![Python](https://img.shields.io/pypi/pyversions/finite-selection)](https://pypi.org/project/finite-selection/)

The Finite Selection Model (FSM) in Python: randomized assignment of units to
treatment groups that balances covariates by design. It reproduces the R package
[FSM](https://cran.r-project.org/package=FSM) to the last rounded decimal.

Carl Morris designed the FSM for the RAND Health Insurance Experiment. Treatment
groups take turns choosing units, like captains picking teams. The order of
turns is random but fair: no group gets far ahead. On its turn, a group takes
the remaining unit that most improves its own design. By default this is the
unit that most increases the determinant of the group's X'X (D-optimality).
Every group pursues the same goal from a random start, so the groups end up
alike. Because the turn order is random, randomization-based inference still
applies.

## Installation

```bash
pip install finite-selection
pip install "finite-selection[data,plot]"   # example data and love plots
```

## Example

Split the 445 men of the National Supported Work data into two groups. Then
compare covariate balance with a completely randomized design. Balance is
measured by TASMD, the distance between a group's covariate means and the
full-sample means, in standard deviations.

```python
import numpy as np
import finite_selection as fs

x = fs.lalonde()
rng = np.random.default_rng(2026)

order = fs.som([222, 223], method="scomars", rng=rng)  # who chooses when
fsm_z = fs.fsm(x, order, rng=rng).assignment  # who goes where
crd_z = fs.crd([222, 223], rng=rng)  # complete randomization

print(fs.tasmd_rand(x, {"CRD": crd_z, "FSM": fsm_z}).round(3))
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

The package also covers:
- Sequential batches: `fsm_batch`.
- Other selection functions: A-optimality, principal-component and average
  based.
- Fisher randomization tests: `perm_test`.
- Effective sample sizes for comparing designs: `ess_rand`, `ess_model`.
- Squares and interactions for balancing beyond the means: `make_sq_inter`.
- Love plots: `love_plot`.

## Parity with R

`tests/fixtures/make_r_fixtures.R` records the outputs of R's exported functions
over a grid of cases, and the test suite checks Python against them.

- **Exact agreement** given the same selection order, on 67 `fsm` cases and 3
  `fsm_batch` cases: the same allocation, the same selection sequence, and
  criteria equal to R's seventh decimal. These cover all 11 selection
  functions, 2 to 4 groups, and options such as no intercept, raw scales,
  ε, `Q_initial` and collinear data.
- **Distributional agreement** for the random draws, against 10,000 R draws
  each. R and numpy use different random number generators, so draws can't
  match one for one.
- **Departures from R.** In a few places Python follows the documented
  behaviour rather than R 1.0.0. For example, `ess_rand` reads only the
  first N assignments in R, and `perm_test` does not impute the null when
  τ ≠ 0. These are listed, with tests, in
  [Behaviour and parity with R](https://finite-sample.github.io/finite-selection/behavior.html).

## Credits

The FSM is due to Carl N. Morris (1979, *Journal of Econometrics* 11, 43–61).
Ambarish Chattopadhyay, Carl N. Morris and José R. Zubizarreta developed it
for the potential-outcomes framework and wrote the R package
[FSM](https://cran.r-project.org/package=FSM):

- "Randomized and balanced allocation of units into treatment groups using
  the finite selection model for R" (arXiv:2105.02393).
- "Balanced and robust randomized treatment assignments: the finite selection
  model for the Health Insurance Experiment and beyond", *JRSS-A* 189(4),
  2026 (arXiv:2205.09736).

This package is an independent implementation, checked against their R
package. The bundled data is Dehejia and Wahba's subset of the National
Supported Work demonstration (LaLonde 1986), from
[NBER](https://users.nber.org/~rdehejia/nswdata.html).

Please cite the papers above when you use the method, and this package (see
`CITATION.cff`) when you use the software.
