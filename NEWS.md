# StateSpaceEcon.jl release notes

## v0.7.0 - Symbolics-based core

This release rebuilds the solver internals on the Symbolics.jl-based
`ModelBaseEcon` core (v0.8). The high-level solver API is preserved -
steady-state solve, stacked-time simulation, first-order solve, the
Kalman filter/smoother, shock decomposition, stochastic simulation, and
the DFM EM estimator are all carried forward, with legacy names kept as
thin compatibility aliases. What changed is the implementation: residuals
and Jacobians are now assembled from the exportable equation kernels the
new core generates, sparsity is precomputed at plan construction, and the
solvers run a single pluggable-linsolve Newton path. The result is faster
model builds and simulations with bit-identical first-order, Kalman, and
shock-decomposition output against the prior release.

### Data model: `SimData` is a plain labelled matrix

`SimData` is now a labelled `(time x column)` `Matrix{Float64}` over a
unified `[vars; shocks]` column space, in solver space (a `@log` column
stores `log(level)`; use `level_value` / `set_level!` to read and write
levels). The `MVTSeries` surface is provided by a `TimeSeriesEcon`
package extension: `using TimeSeriesEcon` enables wrapping a `SimData`
as an `MVTSeries` view, adopting an `MVTSeries` into a `SimData`, and
loading a longbase CSV directly into an `MVTSeries`. Without
`TimeSeriesEcon` loaded, the matrix API is primary and self-contained.

### `Plan` and the simulation entry points

The simulation plan type is `SimPlan` (carrying the slot maps, sparse
Jacobian, parameter values, and the exogenous/endogenous column mask).
`Plan` is kept as an alias. Exogenize/endogenize and autoexogenize round
trips are expressed with `exogenize!` / `endogenize!` /
`autoexogenize_plan!`.

### Breaking change: eager parameter links

Inherited from the `ModelBaseEcon` core: parameter `@link`s are resolved
eagerly at `@initialize`, so a model simulated here carries fully
resolved parameters. Runtime-resolved cross-model links (satellite
models, `@replaceparameterlinks`) are no longer supported. Within a
single model, `@link` works as before. See the `ModelBaseEcon` release
notes for the full migration.

### Intentional non-ports

The following legacy capabilities are not carried by this release. They
are real capabilities (not obsolete), documented as deliberate non-ports;
the numerical core each one sat on is fully covered. They can be
transcribed later if a consumer needs them.

- **Steady-state presolve** - the coordinate-wise 1D Newton + bisection
  presolve. The steady-state solver uses a single Newton /
  Gauss-Newton-least-squares path, which covers the tutorial set.
- **Stacked-time `sim_lm` / `sim_gn`** - the Levenberg-Marquardt and
  Gauss-Newton Newton variants. The one Newton path covers the tutorial
  set; the linear solve is pluggable (`linsolve=:umfpack` default,
  `:pardiso` via the Pardiso extension).
- **First-order swapped-plan / shock-back-out** - the autoexogenize
  round-trip masked first-order solve.
- **TimeSeriesEcon-typed DFM convenience surface** - `Plan(dfm, rng)` /
  `simulate(dfm, ...)` / `kfd2data`. The DFM numerical core (EM, the
  Kalman filter/smoother, the shocks sampler, the impute helpers) is
  fully covered.

### Removed conveniences

Display and comparison utilities with no numerical role were dropped:
`compare_plans`, `printmatrix`, and the plan/data overlay helpers.
