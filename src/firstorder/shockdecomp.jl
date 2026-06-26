##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# First-order shock decomposition.
#
# Runs the same per-period first-order recursion as `first_order_simulate`,
# but alongside the shocked path it carries a *contribution* matrix that
# splits each endogenous response into the source that produced it:
# initial conditions (`:init`), each shock, and a `:nonlinear` residual
# (≈ 0 for a linear model). This is a different algorithm and output
# structure from the stacked-time `shock_decomp.jl` - no code is shared.
#
# Decomposition recursion: where the shocked path solves
#     sol_t = MAT_n \ (RHS - MAT_x * e_t),
# the contributions matrix SD (rows = endog unknowns, cols = sources) is
# propagated by the same linear operator:
#     SD_RHS  = RbyZbb * SD_{t-1}              (state contributions roll forward)
#     SD_RHS[:, shock_cols] -= MAT_x * diag(e_t - control_t)   (new shock input)
#     SD_t    = MAT_n \ SD_RHS
# Initial conditions seed the `:init` column. By linearity the per-source
# columns sum to the total shocked-control deviation; `:nonlinear` captures
# whatever residual the linear approximation leaves (zero here).
# ----------------------------------------------------------------------

"""
Result of a first-order shock decomposition. `source_names` lists the
contribution columns in order `[:init, shocks..., :nonlinear]`. `contrib`
maps each endogenous variable name to a `(maxlag + T + maxlead) x nsource`
matrix; row `maxlag + t` holds period-`t` contributions. `shocked` and
`control` are the two solver-space simulation arrays (level space after the
caller's boundary conversion). For every variable `v` and period `t`,
`sum(contrib[v][maxlag+t, :]) ≈ shocked[v_col][maxlag+t] - control[...]`.
"""
struct FirstOrderShockDecompResult
    source_names::Vector{Symbol}
    contrib::Dict{Symbol, Matrix{Float64}}
    control::Matrix{Float64}
    shocked::Matrix{Float64}
end

"""
    first_order_shockdecomp(fom, shocked_exog, control; ) -> FirstOrderShockDecompResult

Decompose the first-order simulation of `shocked_exog` (a solver-space,
`[vars; shocks]`-layout matrix as in `first_order_simulate`) relative to the
`control` array (same layout, the baseline solution). Both arrays carry the
initial conditions in their first `maxlag` rows and the period-by-period
exogenous data in the simulation rows.
"""
function first_order_shockdecomp(fom::FirstOrderModel,
                                 shocked_exog::AbstractMatrix{Float64},
                                 control::AbstractMatrix{Float64})
    vm = fom.vm
    maxlag = fom.maxlag
    n_rows = size(shocked_exog, 1)
    size(control) == size(shocked_exog) ||
        error("first_order_shockdecomp: control and shocked arrays must match in size")

    # the shocked solution (level/solver space) via the standard recursion
    shocked_sol = first_order_simulate(fom, collect(Float64, shocked_exog))

    # deviations from steady state (solver space)
    s_dev = shocked_exog .- reshape(fom.x_ss, 1, :)
    c_dev = control .- reshape(fom.x_ss, 1, :)

    nbck = vm.nbck
    ibck = 1:nbck
    ien  = 1:vm.oex
    iex  = vm.oex .+ (1:vm.nex)
    nendo = vm.nbck + vm.nfwd
    nex = vm.nex

    # source columns: [:init, ex_vars at offset 0 (= shock names)..., :nonlinear]
    shock_sources = [v for (v, t) in vm.ex_vars if t == 0]
    source_names = Symbol[:init, shock_sources..., :nonlinear]
    nsrc = length(source_names)
    # map each ex slot to its shock-source column (1-based among shock sources)
    shock_col_of = Dict{Symbol, Int}(s => i for (i, s) in enumerate(shock_sources))

    # per-variable contribution matrices, seeded to zero
    contrib = Dict{Symbol, Matrix{Float64}}()
    for v in fom.model.defs.vars
        contrib[v.name] = zeros(n_rows, nsrc)
    end

    # running contributions: rows = endog unknowns, cols = 1 (init) + nex (shocks)
    ncol = 1 + nex
    SD_t   = zeros(nendo, ncol)
    SD_RHS = zeros(nendo, ncol)
    SD_EX  = zeros(nex, nex)

    sol_t = zeros(nendo + nex)
    RHS   = zeros(nendo)

    # seed initial conditions into the :init column of contrib and SD_t
    tnow0 = maxlag
    for ind in ibck
        (gi, tt) = vm.inds_map[ind]
        sol_t[ind] = s_dev[tnow0 + tt, gi]
        # init contribution = shocked - control at the initial period
        SD_t[ind, 1] = s_dev[tnow0 + tt, gi] - c_dev[tnow0 + tt, gi]
    end
    # record the init seeds in the per-variable matrices (rows 1:maxlag)
    for v in fom.model.defs.vars
        gi = vm.vi[v.name]
        for r in 1:maxlag
            contrib[v.name][r, 1] = s_dev[r, gi] - c_dev[r, gi]
        end
    end

    magic = float(nbck > 0)

    for tnow in (maxlag + 1):n_rows
        if nbck > 0
            LinearAlgebra.mul!(RHS, fom.RbyZbb, view(sol_t, ibck))
            LinearAlgebra.mul!(SD_RHS, fom.RbyZbb, view(SD_t, ibck, :))
        else
            fill!(RHS, 0.0)
            fill!(SD_RHS, 0.0)
        end

        # exogenous data for this period (shocked path + per-shock delta)
        fill!(SD_EX, 0.0)
        for (k, ind, (gi, tt)) in zip(1:nex, iex, view(vm.inds_map, iex))
            sol_t[ind] = s_dev[tnow + tt, gi]
            SD_EX[k, k] = s_dev[tnow + tt, gi] - c_dev[tnow + tt, gi]
        end

        # RHS from exogenous: shocked path
        if nbck > 0
            RHS .-= fom.MAT_x * view(sol_t, iex)
        else
            RHS .= .-(fom.MAT_x * view(sol_t, iex))
        end
        # contributions: scatter the per-shock deltas into the shock columns
        SD_RHS[:, 2:ncol] .= (magic .* SD_RHS[:, 2:ncol]) .- fom.MAT_x * SD_EX

        # solve
        sol_t[ien] = fom.MAT_n \ RHS
        SD_t = fom.MAT_n \ SD_RHS

        # scatter contemporaneous contributions into the per-variable matrices
        for v in fom.model.defs.vars
            name = v.name
            gi = vm.vi[name]
            solind = get(vm.bck_inds, (name, 0), -1)
            solind < 0 && (solind = get(vm.fwd_inds, (name, 0), -1))
            solind < 0 && continue
            contrib[name][tnow, 1] = SD_t[solind, 1]
            for s in shock_sources
                contrib[name][tnow, 1 + shock_col_of[s]] = SD_t[solind, 1 + _ex_slot(vm, s)]
            end
        end
    end

    # nonlinear column = total deviation - sum of attributed sources
    for v in fom.model.defs.vars
        name = v.name
        gi = vm.vi[name]
        M = contrib[name]
        for r in 1:n_rows
            total = shocked_sol[r, gi] - control[r, gi]
            attributed = 0.0
            for c in 1:(nsrc - 1)
                attributed += M[r, c]
            end
            M[r, nsrc] = total - attributed
        end
    end

    return FirstOrderShockDecompResult(source_names, contrib,
                                       collect(Float64, control), shocked_sol)
end

# first matching ex-slot index (among the nex ex_vars) for a shock name at
# offset 0. The shock contribution at offset 0 is the one we attribute.
function _ex_slot(vm::VarMaps, name::Symbol)
    i = get(vm.ex_inds, (name, 0), -1)
    i > 0 || error("first_order_shockdecomp: shock $name has no offset-0 ex slot")
    return i
end
