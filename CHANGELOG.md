# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Set the release version with `uv version X.Y.Z`, record the release here, and
tag the merged commit `vX.Y.Z`. Pushing the matching tag builds and publishes
that version.

## [Unreleased]

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

[Unreleased]: https://github.com/finite-sample/finite-selection/commits/main
