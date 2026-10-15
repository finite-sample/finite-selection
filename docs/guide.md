# User guide

This guide runs one experiment from start to finish on the bundled National
Supported Work data: assign 445 men to two groups, check the balance, compare
the design with complete randomization, and test for an effect. Every output
below comes from running the code shown.

## Assign units

An FSM run takes two inputs: the covariates and a selection order matrix (SOM),
which says which group chooses at each turn. `som` draws the order at random.
For two groups the default method, SCOMARS, keeps each group's count of turns
within one of its expected share at every stage.

```python
import numpy as np
import finite_selection as fs

x = fs.lalonde()
rng = np.random.default_rng(42)

order = fs.som([222, 223], method="scomars", rng=rng)
print(order.treat[:10])
result = fs.fsm(x, order, rng=rng)
print(np.bincount(result.assignment)[1:])
```

```text
[1 2 1 2 2 1 1 2 2 1]
[222 223]
```

`result.assignment` holds each unit's group label (1 or 2), in the row order of
`x`. `result.selected` lists the units in the order they were chosen.

## Check balance

One allocation says little about a design. Draw many, and compare them with
complete randomization (`crd`) on TASMD: the distance between a group's
covariate means and the full-sample means, in standard deviations.

```python
fsm_draws = np.array(
    [fs.fsm(x, fs.som([222, 223], rng=rng), rng=rng).assignment for _ in range(200)]
)
crd_draws = np.array([fs.crd([222, 223], rng=rng) for _ in range(200)])

balance = fs.tasmd_rand(x, {"CRD": crd_draws, "FSM": fsm_draws})
print(balance.mean.round(3))
```

```text
             CRD    FSM
Age        0.039  0.005
Education  0.035  0.006
Black      0.036  0.007
Hispanic   0.037  0.008
Married    0.038  0.006
Nodegree   0.036  0.005
Re74       0.037  0.004
Re75       0.038  0.006
```

On every covariate the FSM's average imbalance is five to nine times smaller.
`balance.sd` gives the spread across draws.

The FSM balances the columns it is given. Here it also balances the squares of
the continuous covariates, which it was never shown:

```python
squares = fs.make_sq_inter(
    x[["Age", "Education", "Re74", "Re75"]], is_inter=False, keep_marginal=False
)
print(fs.tasmd_rand(squares, {"CRD": crd_draws, "FSM": fsm_draws}).mean.round(3))
```

```text
               CRD    FSM
Age^2        0.039  0.004
Education^2  0.035  0.007
Re74^2       0.037  0.003
Re75^2       0.041  0.003
```

That spillover is not guaranteed for every function of the covariates. To
balance squares or interactions on purpose, pass the output of `make_sq_inter`
to `fsm` along with the original columns.

## Compare designs by effective sample size

Balance matters because it shrinks the variance of the effect estimate. The
effective sample size (ESS) puts that in units of sample size: a design's ESS is
N times the smallest variance among the designs compared, divided by its own
variance. To compute it you need potential outcomes, so this example simulates
them. The outcome depends on age, education and 1975 earnings, and treatment
adds 1,000 to every unit.

```python
signal = 50 * x["Age"] + 300 * x["Education"] + 0.3 * x["Re75"]
y_control = (1000 + signal + rng.normal(0, 1000, len(x))).to_numpy()
y_mat = np.column_stack([y_control + 1000, y_control])  # treatment adds 1000

print(fs.ess_rand([crd_draws, fsm_draws], y_mat).round(0))
```

```text
[264. 445.]
```

Complete randomization estimates the effect as precisely as the FSM would with
264 units instead of 445. The gain grows with how well the balanced covariates
predict the outcome, and it vanishes when they predict nothing.

## Test for an effect

The FSM's turn order is random, so the reference distribution for a Fisher
randomization test is the set of allocations the same procedure produces.
`perm_test` takes the observed allocation, the observed outcomes and those
reference draws.

```python
z = fsm_draws[0]
y_obs = np.where(z == 1, y_mat[:, 0], y_mat[:, 1])
test = fs.perm_test(y_obs, z, fsm_draws[1:])
print(round(test.stat_obs), test.p_value)
```

```text
987 0.005
```

The estimate is 987 against a true effect of 1,000. The 199 independent reference
draws exclude the allocation used as the
observed experiment. The Monte Carlo correction includes the observed
statistic once, so the smallest possible p-value is 1/200. More draws give
a finer p-value. With the default two-group contrast and `method="mean"`,
`tau=` tests the sharp null that every unit's effect equals `tau`.

Use `exact=True` only when supplying the complete, equally weighted assignment
distribution, including the observed assignment. A list of distinct FSM
allocations need not have equal probabilities. Rank tests require `tau=0`
and a contrast summing to zero. For more than two groups, the supported null
is that each unit has the same outcome under every treatment, which is stronger
than a zero average treatment effect or a single zero contrast.

## Allocate in batches

When units arrive in batches, `fsm_batch` assigns each new batch given the
earlier ones. Every group starts from the information its earlier members
carry, so the batches together behave like one FSM run.

```python
first, second = x.iloc[:300], x.iloc[300:]
z_first = fs.fsm(first, fs.som([150, 150], rng=rng), rng=rng).assignment
z_second = fs.fsm_batch(
    second, first, z_first, fs.som([72, 73], rng=rng), rng=rng
).assignment
print(np.bincount(np.concatenate([z_first, z_second]))[1:])
```

```text
[222 223]
```

## What the design does not protect against

The FSM balances what it can see. Better balance on observed covariates does
not guarantee smaller estimation variance for every outcome: components that
the design does not balance can fare worse than under complete randomization.
[Fredrik Sävje's discussion](https://doi.org/10.1093/jrsssa/qnag030) examines
this robustness concern. Choose covariates that are relevant to the outcome,
assess plausible outcome models before assignment, and use an analysis that
respects the assignment design.
