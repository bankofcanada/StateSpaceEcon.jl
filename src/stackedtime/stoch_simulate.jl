##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

module StochSimulate

# ----------------------------------------------------------------------
# Stochastic (Monte-Carlo / multi-path) simulation on the unified-column
# SimData/SimPlan surface. This is a *wrapper* around
# `Plans.plan_simulate!` - it adds no new Newton kernel.
#
# Semantics: UNANTICIPATED shocks. A baseline (control) path - typically
# the steady state spread over the plan range, or any deterministic
# anticipated solution - is perturbed, per realisation, by innovations
# that are revealed one period at a time. The recursion is:
#
#   for each period t in the shock range (ascending):
#       build a sub-plan over [t-maxlag .. last], solve it forward with the
#       innovation at t injected into each path's running result, which is a
#       view into the accumulated path matrix so the next period's solve
#       starts from this period's solution.
#
# Because the agent re-solves from t forward knowing only the shocks up to
# and including t, future innovations do not move earlier periods - that is
# exactly what "unanticipated" means and is the property the timing test
# enforces.
#
# Shock API: a 3D matrix `shocks[t, shock, path]` over the model's shock
# columns, where `t` is the 1-based interior simulation period.
# `shock_start` shifts the first shock row to a later interior period so a
# shock range can start mid-horizon.
# ----------------------------------------------------------------------

using ..Plans: SimData, SimPlan, plan_simulate!
using ModelBaseEcon: CompiledModel

export stoch_simulate, StochResult, SimFailed, isfailed

# ----------------------------------------------------------------------
# Failed-path marker.
# ----------------------------------------------------------------------
"""
    SimFailed(period)

Marks a realisation whose Newton solve failed to converge at simulation
period `period`. Stored in place of that path's `SimData` in the result.
"""
struct SimFailed
    period::Int
end

isfailed(::SimFailed) = true
isfailed(::SimData) = false

# ----------------------------------------------------------------------
# Result container.
# ----------------------------------------------------------------------
"""
    StochResult

Holds the per-path results of `stoch_simulate`. Index it like a vector
(`res[p]`) to get path `p`'s `SimData` (or a `SimFailed` marker). Fields:

- `paths :: Vector{Union{SimData, SimFailed}}` - one entry per realisation.
- `n_path :: Int`
- `shock_start :: Int` - interior period of the first shock row.
- `n_shock_period :: Int` - number of shock rows (the shock range length).
"""
struct StochResult
    paths::Vector{Union{SimData, SimFailed}}
    n_path::Int
    shock_start::Int
    n_shock_period::Int
end

Base.length(r::StochResult) = r.n_path
Base.getindex(r::StochResult, i::Int) = r.paths[i]
Base.iterate(r::StochResult, s=1) = s > r.n_path ? nothing : (r.paths[s], s + 1)
Base.eachindex(r::StochResult) = Base.OneTo(r.n_path)
nfailed(r::StochResult) = count(isfailed, r.paths)

# ----------------------------------------------------------------------
# Sub-window solve with copy-in / copy-out.
#
# A sub-plan that starts at interior period `t` and runs to the horizon is
# the faithful unanticipated step: periods before `t` are the sub-plan's
# fixed lag/initial rows, so injecting the shock at `t` cannot propagate
# backward through a lead reference (that is exactly what "unanticipated"
# forbids). We materialise a dense `SimData` over the window
# `[t-maxlag .. T+maxlead]` of the parent, solve it, then copy the solved
# interior rows back into the parent so the next period builds on this one.
#
# (We copy rather than `view` because `SimData.values` is a concrete
# `Matrix{Float64}`; a `SubArray` would be densified on construction,
# silently breaking write-through. Copy-in/out keeps the field concrete -
# and type-stable in the Newton hot loop - at the cost of one window-sized
# copy per period.)
#
# Returns `(converged::Bool)`; on success the parent's interior rows
# `t .. T` carry the updated solution.
# ----------------------------------------------------------------------
function _solve_subwindow!(parent::SimData, model, first_interior::Int,
                           T_sub::Int; tol, maxiter, verbose, linsolve)
    maxlag = parent.maxlag
    maxlead = parent.maxlead
    nrow_sub = maxlag + T_sub + maxlead
    # Parent matrix rows for this window: lag rows `first_interior .. ` map so
    # that parent interior period `first_interior` is the sub's interior 1.
    # Parent interior `t` lives at parent row `maxlag + t`; the sub's lag row 1
    # is parent row `maxlag + first_interior - maxlag = first_interior`.
    r0 = first_interior
    rows = r0:(r0 + nrow_sub - 1)

    sub = SimData(model, maxlag, T_sub; maxlead = maxlead)
    # copy-in
    @inbounds copyto!(sub.values, @view parent.values[rows, :])

    subplan = SimPlan(model, T_sub)            # default mask: vars unknown
    _, conv, _ = plan_simulate!(subplan, sub;
                                tol = tol, maxiter = maxiter,
                                verbose = verbose, linsolve = linsolve)
    conv || return false

    # copy-out: write the solved interior + terminal rows back into the parent
    # (lag rows are unchanged; copying them back is harmless and keeps it simple).
    @inbounds copyto!((@view parent.values[rows, :]), sub.values)
    return true
end

# ----------------------------------------------------------------------
# stoch_simulate
# ----------------------------------------------------------------------
"""
    stoch_simulate(model, baseline::SimData, shocks::Array{Float64,3};
                   shock_start=1, tol=1e-9, maxiter=50, verbose=false,
                   linsolve=:umfpack) -> StochResult

Run `size(shocks, 3)` unanticipated stochastic simulations of `model`
about the `baseline` (control) path.

- `baseline` is the anticipated solution spanning the full plan range
  (`maxlag` lag rows + `T` interior rows + `maxlead` terminal rows), in
  solver space. It is typically the steady state spread over the range.
- `shocks[t, k, p]` is the innovation added to shock column `k` at interior
  period `shock_start + t - 1` for realisation `p`. Shocks are injected
  unanticipated: revealed at their period, not before.
- Each realisation re-solves the stacked-time system forward from each shock
  period, accumulating the response - so the result is the baseline plus the
  cumulative effect of that path's innovations.

Returns a [`StochResult`](@ref); `res[p]` is path `p`'s `SimData` or a
[`SimFailed`](@ref) marker if its Newton solve diverged.

`stoch_simulate` is embarrassingly parallel across paths (each writes its own
`SimData`, no shared mutable state) and may be wrapped in `Threads.@spawn`;
the current implementation runs them sequentially. The loop is path-outer, so
the per-period sparse-Jacobian structure is rebuilt for every (path, period).
Reordering to period-outer with shared sparsity is the obvious thing to try
first if large-fleet stochastic runs ever become a bottleneck.
"""
function stoch_simulate(model::CompiledModel, baseline::SimData,
                        shocks::Array{Float64,3};
                        shock_start::Int = 1,
                        tol::Float64 = 1e-9,
                        maxiter::Int = 50,
                        verbose::Bool = false,
                        linsolve::Symbol = :umfpack)
    baseline.model === model ||
        error("stoch_simulate: baseline and model differ")

    T_shk, n_shk_col, n_path = size(shocks)
    T = baseline.T

    n_shk_col == baseline.n_shock ||
        error("stoch_simulate: shocks has $(n_shk_col) shock columns, " *
              "model has $(baseline.n_shock)")
    shock_start >= 1 ||
        error("stoch_simulate: shock_start must be >= 1")
    shock_start + T_shk - 1 <= T ||
        error("stoch_simulate: shock range [$(shock_start) .. " *
              "$(shock_start + T_shk - 1)] exceeds the baseline horizon T=$T")

    # Early return: no realisations requested.
    if n_path == 0
        return StochResult(Union{SimData,SimFailed}[], 0, shock_start, T_shk)
    end
    # Early return: empty shock range => every path is the baseline.
    if T_shk == 0
        paths = Union{SimData,SimFailed}[copy(baseline) for _ in 1:n_path]
        return StochResult(paths, n_path, shock_start, 0)
    end

    n_var = baseline.n_var
    paths = Vector{Union{SimData,SimFailed}}(undef, n_path)

    for p in 1:n_path
        # Each path starts from the baseline (control).
        path = copy(baseline)
        failed = false

        # Unanticipated recursion: reveal one shock period at a time.
        for i in 1:T_shk
            t = shock_start + i - 1               # interior period of this shock
            # Inject this period's unanticipated innovation into the running
            # result at interior period t (shock columns are n_var+1 .. end).
            for k in 1:n_shk_col
                path.values[path.maxlag + t, n_var + k] += shocks[i, k, p]
            end

            # Sub-window: solve [t .. T] forward (T_sub interior periods),
            # copying the solution back into the accumulated path.
            T_sub = T - t + 1
            ok = try
                _solve_subwindow!(path, model, t, T_sub;
                                  tol = tol, maxiter = maxiter,
                                  verbose = verbose, linsolve = linsolve)
            catch
                false
            end
            if !ok
                paths[p] = SimFailed(t)
                failed = true
                break
            end
        end

        failed || (paths[p] = path)
    end

    return StochResult(paths, n_path, shock_start, T_shk)
end

end # module StochSimulate
