##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# First-order simulation.
#
# The recursion works in deviation from the steady state. Per period t:
#   RHS   = RbyZbb · bck_{t-1}                       (contribution of the state)
#   sol_t = MAT_n \ (RHS − MAT_x · e_t)              (solve for the unknowns)
# and the `(var, 0)` entries of sol_t are written back to the output.
#
# The simulation array `values` is in solver space and the unified
# `[vars; shocks]` column layout (identical to SimData.values), with
# `maxlag` initial-condition rows, then T simulation rows, then `maxlead`
# terminal rows. `@log` columns are already log(level) in solver space, and
# `x_ss` is likewise in solver space, so the deviation is a plain subtraction
# - no transform is needed inside the recursion. Callers convert to/from
# levels at the boundary (e.g. via SimData's `level_value` / `set_level!`).
#
# This is the "empty plan" path (all variables endogenous, all shocks
# exogenous data) - the headline first-order simulation. Shock back-out
# (the swapped plan) is handled by the autoexogenize variant below.
# ----------------------------------------------------------------------

"""
    first_order_simulate(fom::FirstOrderModel, values::AbstractMatrix) -> Matrix

Simulate the first-order model forward. `values` is a `(maxlag + T + maxlead)
× n_col` matrix in solver space and `[vars; shocks]` column order: rows
`1:maxlag` are initial conditions, rows `maxlag+1 : maxlag+T` carry the
exogenous shock data for the simulation periods, and (when `maxlead > 0`)
the trailing rows hold the terminal exogenous data. Returns a new matrix of
the same shape with the variable columns filled in for the simulation rows.

The recursion is deterministic / unanticipated (`anticipate=false`).
"""
function first_order_simulate(fom::FirstOrderModel, values::AbstractMatrix{Float64})
    vm = fom.vm
    maxlag = fom.maxlag
    n_rows = size(values, 1)

    # work in deviation from the steady state
    dev = Matrix{Float64}(undef, size(values))
    @inbounds for c in 1:size(values, 2), r in 1:n_rows
        dev[r, c] = values[r, c] - fom.x_ss[c]
    end
    sim = copy(dev)

    nbck = vm.nbck
    ibck = 1:nbck
    ien  = 1:vm.oex
    iex  = vm.oex .+ (1:vm.nex)

    sol_t = zeros(vm.nbck + vm.nfwd + vm.nex)
    RHS   = zeros(vm.nbck + vm.nfwd)

    # initial conditions: only the bck entries are used (sol_t[ibck]).
    # tnow is the row index of the current period; period-1 row is maxlag+1.
    tnow0 = maxlag
    for ind in ibck
        (gi, tt) = vm.inds_map[ind]
        sol_t[ind] = dev[tnow0 + tt, gi]
    end

    sol_bck = view(sol_t, ibck)

    for tnow in (maxlag + 1):n_rows
        if nbck > 0
            LinearAlgebra.mul!(RHS, fom.RbyZbb, sol_bck)
        else
            fill!(RHS, 0.0)
        end

        # fill exogenous data for this period
        for (ind, (gi, tt)) in zip(iex, view(vm.inds_map, iex))
            sol_t[ind] = dev[tnow + tt, gi]
        end

        # solve for the endogenous unknowns
        if nbck > 0
            sol_t[ien] = fom.MAT_n \ (RHS - fom.MAT_x * sol_t[iex])
        else
            sol_t[ien] = fom.MAT_n \ (-(fom.MAT_x * sol_t[iex]))
        end

        # write the (var, 0) entries back to the output (bck preferred for
        # mixed variables, then fwd, then ex - same priority as legacy).
        _scatter_period!(sim, tnow, sol_t, vm)
    end

    # add the steady state back
    out = sim
    @inbounds for c in 1:size(out, 2), r in 1:n_rows
        out[r, c] += fom.x_ss[c]
    end
    return out
end

# write sol_t's contemporaneous entries into row `tnow` of `sim`.
function _scatter_period!(sim::AbstractMatrix{Float64}, tnow::Int,
                          sol_t::AbstractVector{Float64}, vm::VarMaps)
    @inbounds for (name, gi) in vm.vi
        solind = get(vm.bck_inds, (name, 0), -1)
        if solind > 0
            sim[tnow, gi] = sol_t[solind]
            continue
        end
        solind = get(vm.fwd_inds, (name, 0), -1)
        if solind > 0
            sim[tnow, gi] = sol_t[solind]
            continue
        end
        solind = get(vm.ex_inds, (name, 0), -1)
        if solind > 0
            sim[tnow, gi] = sol_t[vm.oex + solind]
            continue
        end
        # name not in any contemporaneous class at offset 0 - leave as-is
        # (e.g. a shock that only appears at a nonzero offset). Such a column
        # keeps its exogenous input value, already in `sim` from the copy.
    end
    return nothing
end
