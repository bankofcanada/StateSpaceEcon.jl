# ----------------------------------------------------------------------
# Compatibility surface
#
# Legacy public names kept as thin wrappers over the current solver
# internals, so existing call sites keep working. The wrappers add no new
# capability; each forwards to the present API.
# ----------------------------------------------------------------------

module Compat

using ..Plans: SimPlan, SimData, plan_simulate!, autoexogenize_plan!

export Plan
export simulate, solve!
export use_pardiso, use_umfpack
export FinalCondition, FCNone, FCGiven, FCMatchSSLevel, fclevel, setfc!

# A plan over the unified column space. The previous public spelling was
# `Plan`; it is kept as an alias of `SimPlan`.
const Plan = SimPlan

# ----------------------------------------------------------------------
# Top-level simulation entry points
#
# The previous router took `(model, plan, data)` and dispatched by solver
# name. The current entry is `plan_simulate!(plan, data)`. `simulate`
# is a non-mutating convenience over it (it copies `data` first); `solve!`
# is the in-place spelling.
# ----------------------------------------------------------------------

"""
    simulate(plan::SimPlan, data::SimData; kwargs...) -> SimData

Non-mutating stacked-time simulation: copy `data`, solve in place, and
return the solved copy. Keyword arguments are forwarded to
`plan_simulate!`.
"""
function simulate(plan::SimPlan, data::SimData; kwargs...)
    out = copy(data)
    plan_simulate!(plan, out; kwargs...)
    return out
end

"""
    solve!(plan::SimPlan, data::SimData; kwargs...) -> SimData

In-place stacked-time simulation; forwards to `plan_simulate!` and
returns the mutated `data`.
"""
function solve!(plan::SimPlan, data::SimData; kwargs...)
    plan_simulate!(plan, data; kwargs...)
    return data
end

# ----------------------------------------------------------------------
# Linear-solver selection
#
# The linear solve is selected with the `linsolve` keyword on
# `simulate!` / `plan_simulate!` (`:umfpack` default, `:pardiso` via the
# Pardiso extension). `use_pardiso()` / `use_umfpack()` return the
# matching symbol for call sites that prefer the named-helper spelling.
# ----------------------------------------------------------------------

use_pardiso() = :pardiso
use_umfpack() = :umfpack

# ----------------------------------------------------------------------
# Final (terminal) conditions
#
# Terminal conditions are value-based rows: the trailing `maxlead` rows of
# a `SimData` hold the boundary values the lead references read at the end
# of the horizon. The previous `FCType` constructor/setter names map onto
# writing those rows.
#
# Only the level final conditions are provided here. The rate final
# conditions (matching a steady-state growth rate at the terminal) are
# not part of this release.
# ----------------------------------------------------------------------

abstract type FinalCondition end

"""
    FCNone()

No explicit terminal condition; the trailing rows are left as supplied.
"""
struct FCNone <: FinalCondition end

"""
    FCGiven()

The terminal rows are given values (the default value-based terminal). A
`SimData` already carries these rows; `setfc!` writes them.
"""
struct FCGiven <: FinalCondition end

"""
    FCMatchSSLevel()

The terminal rows match a supplied steady-state level. Equivalent to
`FCGiven` with the steady state written into the terminal rows.
"""
struct FCMatchSSLevel <: FinalCondition end

"""
    fclevel(data::SimData, name) -> Vector{Float64}

Read the terminal-condition rows for column `name` (the trailing
`maxlead` rows, in solver space).
"""
function fclevel(data::SimData, name::Symbol)
    c = data.col_index[name]
    first_term = data.maxlag + data.T + 1
    return data.values[first_term:end, c]
end

"""
    setfc!(data::SimData, ::FinalCondition, name, values)

Write the terminal-condition rows for column `name` (the trailing
`maxlead` rows). `values` may be a scalar (broadcast) or a vector of
length `maxlead`.
"""
function setfc!(data::SimData, ::FCGiven, name::Symbol, values)
    c = data.col_index[name]
    first_term = data.maxlag + data.T + 1
    data.values[first_term:end, c] .= values
    return data
end
setfc!(data::SimData, ::FCMatchSSLevel, name::Symbol, values) =
    setfc!(data, FCGiven(), name, values)
setfc!(data::SimData, ::FCNone, name::Symbol, values) = data

end # module Compat
