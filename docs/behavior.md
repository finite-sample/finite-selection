# Behaviour and parity with R

`finite-selection` reproduces the R package
[FSM](https://cran.r-project.org/package=FSM) 1.0.0 (Chattopadhyay, Morris and
Zubizarreta). The code is written independently, from the published algorithm.
Agreement with R is established black-box: `tests/fixtures/make_r_fixtures.R`
calls R's exported functions on a grid of cases and records the inputs and
outputs, and the test suite checks the Python functions against those records.
The `R parity` workflow regenerates the records from a fresh R install weekly.

This page lists the numerical conventions the parity depends on, and the places
where Python deliberately departs from R.

## What "parity" means here

- **Deterministic results are compared exactly.**
  - Given the same selection order and `ties="smallest"`, `fsm` and `fsm_batch`
    return the same allocation and the same selection sequence as R.
  - Their criterion matrices agree to R's seventh rounded decimal.
  - `perm_test`, `ess_rand`, `ess_model`, `tasmd_rand` and `make_sq_inter`
    agree to 1e-10 or better on identical inputs.
- **Random draws are compared as distributions.**
  - R and numpy use different random number generators, so `som`, `crd` and
    `fsm(ties="random")` cannot match R draw for draw.
  - What is exact is everything downstream of a draw. For example, SCOMARS's
    conditional probabilities along R's own path match exactly.
  - The draws themselves are tested against 10,000 R draws, stage by stage
    or unit by unit.
- **The bundled Lalonde data matches R's `Lalonde`** to 1e-9.
  - It is rebuilt from the public NBER file by `scripts/build_lalonde.py`,
    not copied from R.

## Conventions

The last column names test cases that fail when the convention is changed; each
was checked by mutating it and rerunning the suite.

| Convention | Value | Pinned by |
|---|---|---|
| Standardization | columns centred and divided by the sample SD (n − 1) | 41 cases, e.g. `fsm_*_max_average`, `fsm_*_min_pc` |
| Design matrix | intercept column first, then covariates | all `fsm_*` |
| Ridge | `eps * Q0 / N`, where `Q0` is X'X of the full standardized design (or `q_initial`) and `N` the number of units | `fsm_gauss_eps*`, `fsm_gauss_qinit`, `fsm_collinear_qinit` |
| Empty group | criterion `x' (ridge)^-1 x` | 24 D- and A-optimal cases |
| Regularization switch | if LAPACK's 1-norm reciprocal condition estimate of the R factor of `X_g'X_g / stage` is below 1e-6, use `(X_g'X_g / n_g + ridge)^-1`; otherwise `(X_g'X_g)^-1` | `fsm_gauss_Aopt`, `fsm_*_negative_Dopt`, `batch_lalonde60_k3` |
| Criterion rounding | 7 decimals before the argmax | `fsm_binary_*`, `fsm_ties_random_binary` |
| Ties | `"smallest"`: earliest-listed unit; `"random"`: uniform among maximizers | 45 cases; `fsm_ties_random_binary` |
| `aopt` | `x' S T S x / (1 + x' S x)`, `T = P' W P` | `fsm_*_Aopt` |
| `max_pc`, `min_pc` | first principal component scores of the centred covariates | `fsm_*_max_pc` |
| `dopt_pc`, `dopt_average` | sum of squared deviations of the group plus candidate, on the first PC or the row mean | `fsm_*_Dopt_pc` |
| `marginal_var_sum` | trace of the covariance of the group plus candidate; with a single covariate, the sum of squares instead (same choice, different reported value) | `fsm_ages12_marginal_var_sum` |
| Collinear covariates | an error, unless `q_initial` is given | `fsm_collinear_*` |
| `fsm_batch` | standardization, `Q0` and `N` use past and new units together; each group's X'X starts from its past members | `batch_*`; `N` by `batch_lalonde60_k3` |
| SCOMARS | `P(second group at i+1) = clip((p_{i+1} − max(0, S_i − F_i)) / (1 − abs(S_i − F_i)), 0, 1)` | `som_scomars_*` |
| Global percentage | the group with the smallest `(count + 0.5) / size` chooses next; ties are random | `som_gp_k3` |
| Randomized chunk | each block of `size / gcd(sizes)` turns is a random permutation | `som_chunk_*` |
| ESS (randomization) | `N * min(var) / var` of the difference-in-means across assignments | `ess_rand` |
| ESS (model) | regression imputation at the full-sample covariate mean; residual variance on `n_g − p` degrees of freedom | `ess_model_unequal` |
| TASMD | `abs(group mean − target) / SD`; the target SD defaults to the full-sample SD | `tasmd_rand` (`vec_target`, `mat_target`) |
| `make_sq_inter` names | `a.b` for products, `a^2` for squares; marginal, then products, then squares | `make_sq_inter` |

## Where Python departs from R

Each departure is deliberate and has a test that pins both R's behaviour and
Python's.

1. **`ess_rand` uses every assignment.**
   - R 1.0.0 reads only the first N assignments of each design, N being the
     number of units, and fails when there are fewer.
   - On the first N assignments the two agree exactly.
2. **`perm_test` with `tau ≠ 0` imputes the sharp null.**
   - Under `Y_i(1) − Y_i(2) = tau`, Python fills in each unit's missing
     potential outcome and recomputes the statistic for every reference
     assignment. Two-sided p-values measure distance from `tau`.
   - R keeps the outcomes fixed and subtracts `tau` from every statistic.
     Its one-sided p-values therefore never depend on `tau`.
   - For the default contrast at `tau = 0`, the reference statistics agree.
     General two-group contrasts require imputation even when `tau = 0`.
3. **`perm_test` counts ties.**
   - Statistics such as rank means can tie the observed value exactly while
     differing in the last floating-point bits.
   - R compares raw floats and drops some of these ties. Python counts them,
     when computing the tail probability.
   - R's p-value always lies between the tie-free and the tie-inclusive
     value.
4. **Criteria of exactly −1 are reported.** R marks unavailable units with −1
   before converting to NA, so a genuine criterion of −1 also becomes NA.
5. **`tasmd_rand` accepts designs with different numbers of assignments.** R
   fails with "subscript out of bounds".
6. **Clear errors.**
   - Collinear covariates raise a `ValueError` that suggests `q_initial`,
     where R fails inside `solve()`.
   - `make_sq_inter` with every option off raises, where R fails on an
     undefined object.

## API differences

The Python API follows Python conventions; results are unchanged.

- Names are snake case: `s_function="negative_dopt"`, `method="randomized_chunk"`,
  `alternative="two-sided"`.
- Covariates are arrays or data frames without an index column.
- `FSMResult.selected` holds 0-based row positions.
- `ess_rand` takes a list of per-design arrays (one assignment per row).
  R takes a 3-D array.
- `ess_model` takes one assignment per row; R takes one per column.
- `tasmd_rand` takes a dict of named designs and returns unrounded values.
- `love_plot` draws on matplotlib axes.
  - Its TASMD values are tested against `tasmd`.
  - Its ASMD option divides by the root mean of the two group variances. R
    exposes no ASMD numbers to compare against.

## Randomization-test contract

For two groups, `perm_test` imputes the null
`c_1 Y_i(1) + c_2 Y_i(2) = tau` for every unit, including `tau=0`. Both
coefficients must be nonzero. For more than two groups, `tau=0` with a
nonzero contrast summing to zero tests the stronger sharp no-effect null,
not just the single contrast constraint. Rank statistics also require the
no-effect null; a nonzero effect in outcome units is not a shift in rank units.

By default, reference assignments are independent draws from the experiment's
design. The p-value is `(b + 1) / (B + 1)`, including the observed statistic
once. Do not deliberately add the observed assignment to the sampled rows.
Set `exact=True` only for the full equally weighted reference distribution,
including the observed assignment. The software checks its presence but
cannot establish that a supplied distribution is complete or correctly weighted.
The R fixtures include the observed assignment as their first row; parity
tests pass the remaining rows as sampled references.

`tests/test_perm_test.py` checks null imputation against explicit potential
outcomes, all three alternatives, Monte Carlo correction, outcome rescaling,
and rejection of invalid or unsupported inputs.
