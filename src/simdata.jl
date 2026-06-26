# ----------------------------------------------------------------------
# SimData - a labelled (time x column) data container.
#
# Columns are the unified space `[vars...; shocks...]` in model
# declaration order. Rows span the full plan range including the
# `maxlag` initial-condition rows.
#
# IMPORTANT - log/level convention: `values` holds data in *solver
# space*. For a `@log` variable the codegen makes the solver unknown
# the log of the variable (every equation reference becomes `exp(x)`),
# so a `@log` column stores `log(level)`. Plain variables and shocks
# store the level directly. The `level_*` accessors and `load_longbase`
# convert at the boundary; the raw `[]`/`values` interface is solver
# space and is what `plan_simulate!` reads and writes.
# ----------------------------------------------------------------------

struct SimData
    model::ModelBaseEcon.CompiledModel
    values::Matrix{Float64}              # (maxlag + T + maxlead) x (n_var + n_shock)
    col_index::Dict{Symbol, Int}         # name -> column
    is_log_col::Vector{Bool}             # per column: @log variable?
    n_var::Int
    n_shock::Int
    maxlag::Int
    T::Int
    maxlead::Int                         # trailing terminal-condition rows
end

"""
    SimData(model, maxlag, T; maxlead=0) -> SimData

Zero-initialised data container: `maxlag + T + maxlead` rows, one column
per variable then per shock. Row `maxlag + t` is simulation period `t`;
rows `1..maxlag` hold initial conditions; rows
`maxlag+T+1 .. maxlag+T+maxlead` hold terminal conditions.

For a model with leads (`maxlead > 0`) the terminal rows must be supplied
(typically the steady state) so the lead references at the final
simulation periods read valid boundary data - `plan_simulate!` treats
them as fixed (an `fcgiven`-style terminal). Pass `maxlead = model_maxlead`
(or use `SimPlan(...).maxlead`) when simulating a lead model. The default
`maxlead = 0` is backward-compatible with lead-free models.
"""
function SimData(model::ModelBaseEcon.CompiledModel, maxlag::Int, T::Int;
                 maxlead::Int = 0)
    def = model.defs
    n_var = length(def.vars)
    n_shock = length(def.shocks)
    col_index = Dict{Symbol, Int}()
    is_log_col = falses(n_var + n_shock)
    for (i, v) in pairs(def.vars)
        col_index[v.name] = i
        is_log_col[i] = v.kind === IR.VAR_LOG
    end
    for (i, s) in pairs(def.shocks); col_index[s.name] = n_var + i;    end
    values = zeros(maxlag + T + maxlead, n_var + n_shock)
    return SimData(model, values, col_index, is_log_col,
                   n_var, n_shock, maxlag, T, maxlead)
end

# Raw column access by name (solver space): `sd[:rff]` -> the full
# column vector view; `sd[:rff, t]` -> period-t cell.
Base.getindex(sd::SimData, name::Symbol) =
    view(sd.values, :, sd.col_index[name])
Base.getindex(sd::SimData, name::Symbol, t::Int) =
    sd.values[sd.maxlag + t, sd.col_index[name]]
function Base.setindex!(sd::SimData, v, name::Symbol)
    sd.values[:, sd.col_index[name]] .= v
    return v
end
function Base.setindex!(sd::SimData, v::Real, name::Symbol, t::Int)
    sd.values[sd.maxlag + t, sd.col_index[name]] = v
    return v
end

Base.copy(sd::SimData) = SimData(sd.model, copy(sd.values), sd.col_index,
                                 sd.is_log_col, sd.n_var, sd.n_shock,
                                 sd.maxlag, sd.T, sd.maxlead)

sim_range_length(sd::SimData) = sd.T

"""
    level_value(sd, name, t) -> Float64

Read cell `(name, t)` as a *level*: for a `@log` column the stored value
is `log(level)`, so this returns `exp` of it; otherwise the stored value.
"""
function level_value(sd::SimData, name::Symbol, t::Int)
    c = sd.col_index[name]
    v = sd.values[sd.maxlag + t, c]
    return sd.is_log_col[c] ? exp(v) : v
end

"""
    set_level!(sd, name, t, level)

Write `level` into cell `(name, t)`: for a `@log` column the stored
value is `log(level)`; otherwise `level` directly.
"""
function set_level!(sd::SimData, name::Symbol, t::Int, level::Real)
    c = sd.col_index[name]
    sd.values[sd.maxlag + t, c] = sd.is_log_col[c] ? log(level) : level
    return level
end
