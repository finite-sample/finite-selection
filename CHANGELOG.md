# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Set the release version with `uv version X.Y.Z`, record the release here, and
tag the merged commit `vX.Y.Z`. Pushing the matching tag builds and publishes
that version.

## [Unreleased]

## 0.1.0 - 2026-10-15

### Added

- `fsm` and `fsm_batch`: the Finite Selection Model with all eleven selection
  functions of the R package FSM, reproducing its allocations exactly given the
  same selection order.
- `som` (SCOMARS, global percentage, randomized chunk) and `crd`.
- `perm_test`, `ess_rand`, `ess_model`, `tasmd`, `tasmd_rand`, `make_sq_inter`,
  `asmd` and `love_plot`.
- The Lalonde (Dehejia–Wahba NSW) data, rebuilt from the NBER source.
- R parity fixtures, the suite that checks them, and a weekly workflow that
  regenerates them from a fresh R install.

### Fixed

- Impute missing potential outcomes for general two-group nulls even at zero tau.
- Apply the Monte Carlo p-value correction to sampled reference assignments;
  expose exact enumeration explicitly.
- Reject unsupported rank-test nulls and invalid inference inputs, and clarify
  that multi-group tests use the sharp no-effect null.
- Keep numerical tie handling relative to the observed statistic's scale.

[Unreleased]: https://github.com/finite-sample/finite-selection/commits/main
