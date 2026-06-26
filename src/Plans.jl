module Plans

# ----------------------------------------------------------------------
# Plan-based stacked-time simulation with an exogenous/endogenous swap,
# plus a longbase-style data container.
#
# The base `StackedTimeSolver.simulate!` solves a square system where the
# unknowns are exactly all variable cells and every shock is exogenous
# data. The longbase workflow needs two extensions:
#
#  1. A square system that is NOT n_eq == n_var. A model may have, say,
#     284 equations, 284 endogenous variables, and 367 shock-columns (83
#     `@exogenous` variables + 284 auto-shocks). The per-period system is
#     square in (equations, endogenous variables); the shock columns are
#     all data.
#
#  2. An exog/endo *swap* (`autoexogenize`): to back out shocks from a
#     historical baseline, a set of variables is held fixed to data and
#     the matching shocks become unknowns. Still square.
#
# This module works on a unified column space of `n_var + n_shock`
# columns. A per-column `is_unknown` mask picks which columns the Newton
# step solves for; the count of unknown columns must equal `n_eq`. The
# default mask (all vars unknown, all shocks data) reproduces
# `StackedTimeSolver.simulate!` exactly - this module does not replace it,
# it generalises it.
# ----------------------------------------------------------------------

using ModelBaseEcon
using ModelBaseEcon: IR, Symbolic
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC, sparse, nzrange, rowvals, nonzeros

# Share the pluggable linear-solve hooks defined in StackedTimeSolver, so
# plan_simulate! routes through the same Pardiso path.
using ..StackedTimeSolver: _solve_jacobian, _init_linsolve, _finalize_linsolve!

export SimData, SimPlan, exogenize!, endogenize!, autoexogenize_plan!,
       plan_simulate!, sim_range_length, load_longbase, load_longbase_mvts,
       level_value, set_level!

include("simdata.jl")

# ----------------------------------------------------------------------
# SimPlan - a stacked-time plan over a unified column space with a
# per-column unknown/exogenous mask.
# ----------------------------------------------------------------------

"""
Per-equation slot map into the unified column space. Each entry:
`(col::Int, offset::Int)` - column index in `[vars; shocks]`, time
offset relative to `t`.
"""
const UnifiedSlot = Tuple{Int, Int}
const UnifiedSlotMap = Vector{UnifiedSlot}

mutable struct SimPlan
    model::ModelBaseEcon.CompiledModel
    T::Int
    n_var::Int
    n_shock::Int
    n_col::Int                            # n_var + n_shock
    n_eq::Int
    maxlag::Int
    maxlead::Int
    col_index::Dict{Symbol, Int}
    slot_maps::Vector{UnifiedSlotMap}     # per equation
    param_values::Vector{Float64}
    # Mutable: which columns are unknowns. `is_unknown[c]` true => column c
    # is solved for; false => supplied as data. Default: vars unknown.
    is_unknown::Vector{Bool}
end

"""
    SimPlan(compiled::CompiledModel, T::Int) -> SimPlan

Build a plan with the default partition: every variable is endogenous,
every shock is exogenous data. Use `exogenize!` / `endogenize!` /
`autoexogenize_plan!` to swap before `plan_simulate!`.
"""
function SimPlan(compiled::ModelBaseEcon.CompiledModel, T::Int)
    T >= 1 || error("SimPlan: horizon T must be >= 1")
    def = compiled.defs
    n_var = length(def.vars)
    n_shock = length(def.shocks)
    n_col = n_var + n_shock
    n_eq = length(compiled)

    col_index = Dict{Symbol, Int}()
    for (i, v) in pairs(def.vars);   col_index[v.name] = i;            end
    for (i, s) in pairs(def.shocks); col_index[s.name] = n_var + i;    end

    maxlag = 0
    maxlead = 0
    slot_maps = UnifiedSlotMap[]
    for eqn in compiled.eqns
        sm = UnifiedSlotMap()
        for ref in eqn.tsrefs
            haskey(col_index, ref.name) ||
                error("SimPlan: equation references unknown name `$(ref.name)`")
            push!(sm, (col_index[ref.name], ref.offset))
            maxlag = max(maxlag, -ref.offset)
            maxlead = max(maxlead, ref.offset)
        end
        push!(slot_maps, sm)
    end

    param_values = _resolve_param_values(def, compiled.param_layout)

    # Default mask: variables (columns 1..n_var) are unknowns.
    is_unknown = falses(n_col)
    is_unknown[1:n_var] .= true

    return SimPlan(compiled, T, n_var, n_shock, n_col, n_eq, maxlag, maxlead,
                   col_index, slot_maps, param_values, is_unknown)
end

function _resolve_param_values(def::IR.ModelDef,
                                layout::Vector{Symbolic.ParamRef})
    by_name = Dict{Symbol, IR.ParamDecl}()
    for p in def.params; by_name[p.name] = p; end
    vals = Vector{Float64}(undef, length(layout))
    for (i, ref) in pairs(layout)
        p = by_name[ref.name]
        if p.kind === IR.PARAM_SCALAR
            vals[i] = Float64(p.value)
        elseif p.kind === IR.PARAM_ARRAY
            vals[i] = Float64(p.value[ref.index])
        else
            error("root layout should only hold scalar/array params, got $(p.kind)")
        end
    end
    return vals
end

# ----------------------------------------------------------------------
# Exog/endo swap API
# ----------------------------------------------------------------------

"""
    exogenize!(plan, name)

Make column `name` exogenous (data, not solved for).
"""
function exogenize!(plan::SimPlan, name::Symbol)
    plan.is_unknown[plan.col_index[name]] = false
    return plan
end

"""
    endogenize!(plan, name)

Make column `name` endogenous (an unknown solved for by the Newton step).
"""
function endogenize!(plan::SimPlan, name::Symbol)
    plan.is_unknown[plan.col_index[name]] = true
    return plan
end

"""
    autoexogenize_plan!(plan)

Apply the model's `@autoexogenize` pairs: for each `var => shock`, make
`var` exogenous and `shock` endogenous. This is the swap used to back
shocks out of a historical baseline.
"""
function autoexogenize_plan!(plan::SimPlan)
    for pair in plan.model.defs.autoexog
        exogenize!(plan, pair.var)
        endogenize!(plan, pair.shock)
    end
    return plan
end

# ----------------------------------------------------------------------
# Sparsity + assembly over the unified column space
# ----------------------------------------------------------------------

# Flat index of unknown (t, u) where u is the 1-based index into the
# *list of unknown columns*. Variable-major: (u-1)*T + t.
@inline _uidx(t::Int, u::Int, T::Int) = (u - 1) * T + t

# Build the sparse Jacobian over the current `is_unknown` mask.
# Returns (J, BI, unknown_cols, col_to_unknown).
function _build_sparsity(plan::SimPlan)
    T, n_eq = plan.T, plan.n_eq
    unknown_cols = findall(plan.is_unknown)
    n_unknown = length(unknown_cols)
    n_unknown == n_eq ||
        error("SimPlan: $(n_unknown) unknown columns but $(n_eq) equations - " *
              "the per-period system must be square")
    # Map a global column index -> its position in the unknown list (0 if data).
    col_to_unknown = zeros(Int, plan.n_col)
    for (u, c) in pairs(unknown_cols)
        col_to_unknown[c] = u
    end

    Is = Int[]; Js = Int[]
    block_slots = Vector{Vector{Tuple{Int, Int}}}()
    sizehint!(block_slots, T * n_eq)
    for t in 1:T
        for i in 1:n_eq
            sm = plan.slot_maps[i]
            slots = Vector{Tuple{Int, Int}}()
            for (k, (col, off)) in pairs(sm)
                u = col_to_unknown[col]
                tprime = t + off
                if u != 0 && 1 <= tprime <= T
                    push!(Is, (t - 1) * n_eq + i)
                    push!(Js, _uidx(tprime, u, T))
                    push!(slots, (k, length(Is)))
                else
                    push!(slots, (k, 0))    # boundary or data column
                end
            end
            push!(block_slots, slots)
        end
    end

    n_rows = T * n_eq
    n_cols = T * n_unknown
    J = sparse(Is, Js, ones(Float64, length(Is)), n_rows, n_cols)

    nz_index = Dict{Tuple{Int, Int}, Int}()
    sizehint!(nz_index, length(Is))
    rows = rowvals(J)
    @inbounds for col in 1:n_cols
        for p in nzrange(J, col)
            nz_index[(rows[p], col)] = p
        end
    end

    BI = Vector{Vector{Int}}(undef, T * n_eq)
    for (b, slots) in pairs(block_slots)
        n_slots = isempty(slots) ? 0 : maximum(s -> s[1], slots)
        bi = zeros(Int, n_slots)
        for (k, coo_idx) in slots
            bi[k] = coo_idx == 0 ? 0 : nz_index[(Is[coo_idx], Js[coo_idx])]
        end
        BI[b] = bi
    end
    fill!(nonzeros(J), 0.0)
    return J, BI, unknown_cols, col_to_unknown
end

# Gather an equation's input vector from the unified data matrix.
@inline function _gather(data::Matrix{Float64}, sm::UnifiedSlotMap,
                          t::Int, maxlag::Int)
    n = length(sm)
    x = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        col, off = sm[k]
        x[k] = data[t + off + maxlag, col]
    end
    return x
end

# ----------------------------------------------------------------------
# Newton solve
# ----------------------------------------------------------------------

"""
    plan_simulate!(plan, data::SimData; tol=1e-9, maxiter=50, verbose=false)
        -> (data, converged, iters)

Solve the stacked-time system in place. `data` supplies initial
conditions (rows `1..maxlag`), exogenous columns, and the starting guess
for the unknown columns over the simulation rows. On return the unknown
columns of `data` over rows `maxlag+1 .. maxlag+T` hold the solution.

Which columns are unknown is set by `plan.is_unknown` - by default the
variables; after `autoexogenize_plan!` a swapped set. The number of
unknown columns must equal the number of equations.
"""
function plan_simulate!(plan::SimPlan, data::SimData;
                        tol::Float64 = 1e-9,
                        maxiter::Int = 50,
                        verbose::Bool = false,
                        linsolve::Symbol = :umfpack)
    data.model === plan.model ||
        error("plan_simulate!: data and plan refer to different models")
    # The data matrix must carry the lead model's terminal rows so that lead
    # references at the final simulation periods read valid boundary data.
    size(data.values, 1) == plan.maxlag + plan.T + plan.maxlead ||
        error("plan_simulate!: data has $(size(data.values,1)) rows, " *
              "expected $(plan.maxlag + plan.T + plan.maxlead) " *
              "(maxlag $(plan.maxlag) + T $(plan.T) + maxlead $(plan.maxlead)). " *
              "Construct the SimData with `maxlead = $(plan.maxlead)`.")

    T, n_eq, maxlag = plan.T, plan.n_eq, plan.maxlag
    J, BI, unknown_cols, _ = _build_sparsity(plan)
    M = data.values

    R = Vector{Float64}(undef, T * n_eq)
    grad_buf = Float64[]
    iter = 0
    converged = false
    lv = Val(linsolve)
    lstate = _init_linsolve(lv)
    try
        while iter < maxiter
            iter += 1
            # Assemble residual + Jacobian.
            nz = nonzeros(J)
            fill!(nz, 0.0)
            @inbounds for t in 1:T
                for i in 1:n_eq
                    eqn = plan.model[i]
                    sm = plan.slot_maps[i]
                    if length(grad_buf) != eqn.n_x
                        resize!(grad_buf, eqn.n_x)
                    end
                    x = _gather(M, sm, t, maxlag)
                    R[(t - 1) * n_eq + i] = eqn.eval_RJ!(grad_buf, x,
                                                         plan.param_values)
                    bi = BI[(t - 1) * n_eq + i]
                    for k in 1:length(bi)
                        nz_idx = bi[k]
                        nz_idx == 0 && continue
                        nz[nz_idx] += grad_buf[k]
                    end
                end
            end
            rnorm = norm(R, Inf)
            verbose && @info "plan_simulate iter $iter" rnorm
            if rnorm < tol
                converged = true
                break
            end
            Δ, lstate = _solve_jacobian(lv, J, R, lstate)
            @inbounds for (u, c) in pairs(unknown_cols)
                for t in 1:T
                    M[maxlag + t, c] -= Δ[_uidx(t, u, T)]
                end
            end
            if norm(Δ, Inf) < tol
                # One more residual check after the final step.
                @inbounds for t in 1:T, i in 1:n_eq
                    eqn = plan.model[i]
                    x = _gather(M, plan.slot_maps[i], t, maxlag)
                    R[(t - 1) * n_eq + i] = eqn.eval_resid(x, plan.param_values)
                end
                converged = norm(R, Inf) < tol
                break
            end
        end
    finally
        _finalize_linsolve!(lv, lstate)
    end
    return data, converged, iter
end

include("plandata.jl")

end # module Plans
