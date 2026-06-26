##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Data layout
# ----------------------------------------------------------------------
#
# Suppose the model has n_var declared variables, n_shock declared shocks,
# n_eq equations, and the maximum lag/lead across all equations is
# `maxlag` / `maxlead`.
#
# The simulation horizon is T periods, indexed t = 1..T. The full time
# grid spans 1-maxlag .. T+maxlead, with the first `maxlag` and last
# `maxlead` rows holding boundary data (initial conditions / terminal
# conditions). Internally we use 1-based indexing into a (T+maxlag+
# maxlead) x n_var matrix `x_full`, where `t_int = t + maxlag`.
#
# Unknowns are `x_full[maxlag+1 : maxlag+T, :]` - flattened in
# variable-major order: index of (t, v) is `(v-1)*T + t`. This matches
# the column-major iteration order of a Julia Matrix viewed as a vector.
#
# Shocks are exogenous: caller supplies `e_full` with the same row
# layout but n_shock columns; per-equation shocks are read directly.
# ----------------------------------------------------------------------

"""
Per-equation slot map. Each entry is one of:
- `(:var, vidx::Int, offset::Int)` - gather from `x_full[t+offset, vidx]`
- `(:shock, sidx::Int, offset::Int)` - gather from `e_full[t+offset, sidx]`
"""
const SlotEntry = Tuple{Symbol, Int, Int}
const StackedSlotMap = Vector{SlotEntry}

struct StackedTimePlan
    model::ModelBaseEcon.CompiledModel
    T::Int                                    # simulation horizon
    n_var::Int
    n_shock::Int
    n_eq::Int
    maxlag::Int
    maxlead::Int
    var_index::Dict{Symbol, Int}
    shock_index::Dict{Symbol, Int}
    slot_maps::Vector{StackedSlotMap}         # per equation
    param_values::Vector{Float64}
    # Sparse pattern, built once.
    J::SparseMatrixCSC{Float64, Int}
    # For each (block_index = (t, eq)), the indices into J.nzval that
    # correspond to that equation's gradient slots, in tsref order.
    BI::Vector{Vector{Int}}
end

# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

"""
    StackedTimePlan(compiled::CompiledModel, T::Int)

Precompute slot maps, param values, and the sparse Jacobian pattern for
a stacked-time simulation of horizon `T`.
"""
function StackedTimePlan(compiled::ModelBaseEcon.CompiledModel, T::Int)
    T >= 1 || error("stacked-time horizon T must be >= 1")
    def = compiled.defs
    n_var = length(def.vars)
    n_shock = length(def.shocks)
    n_eq = length(compiled)

    n_eq == n_var ||
        error("stacked-time: square per-period system required (n_eq=$n_eq, n_var=$n_var)")

    var_index = Dict{Symbol, Int}()
    for (i, v) in pairs(def.vars); var_index[v.name] = i; end
    shock_index = Dict{Symbol, Int}()
    for (i, s) in pairs(def.shocks); shock_index[s.name] = i; end

    # Build per-equation slot maps and accumulate global maxlag/maxlead.
    maxlag = 0
    maxlead = 0
    slot_maps = StackedSlotMap[]
    for eqn in compiled.eqns
        sm = StackedSlotMap()
        for ref in eqn.tsrefs
            if haskey(var_index, ref.name)
                push!(sm, (:var, var_index[ref.name], ref.offset))
            elseif haskey(shock_index, ref.name)
                push!(sm, (:shock, shock_index[ref.name], ref.offset))
            else
                error("stacked-time: equation references unknown name `$(ref.name)`")
            end
            maxlag = max(maxlag, -ref.offset)
            maxlead = max(maxlead, ref.offset)
        end
        push!(slot_maps, sm)
    end

    param_values = _resolve_param_values(def, compiled.param_layout)

    # Build sparsity pattern.
    J, BI = _build_sparsity(T, n_var, n_eq, slot_maps)

    return StackedTimePlan(compiled, T, n_var, n_shock, n_eq, maxlag, maxlead,
                            var_index, shock_index, slot_maps, param_values,
                            J, BI)
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

# Global flat index of (t, v). t in 1..T, v in 1..n_var.
@inline _xidx(t::Int, v::Int, T::Int) = (v - 1) * T + t
