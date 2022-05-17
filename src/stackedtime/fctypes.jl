##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    abstract type FinalCondition end

Abstract base type for all final conditions
"""
abstract type FinalCondition end

"""
    struct FCNone <: FinalCondition end

Variables that don't have lags in the model don't need final conditions.
"""
struct FCNone <: FinalCondition end

"""
    struct FCGiven <: FinalCondition end

Final conditions are given in the exogenous data.
"""
struct FCGiven <: FinalCondition end

"""
    struct FCMatchSSLevel <: FinalCondition end

Final conditions match the level of the steady state.
"""
struct FCMatchSSLevel <: FinalCondition end

"""
    struct FCMatchSSRate <: FinalCondition end

Final conditions are such that the solution has the same slope (growth rate or
rate of change) as the steady state solution.
"""
struct FCMatchSSRate <: FinalCondition end

"""
    struct FCConstRate <: FinalCondition end

Final condition is such that the variable grows at a
"""
struct FCConstRate <: FinalCondition end

"""
    const fcnone = FCNone()

Used internally for variables without final conditions, such as shocks and
exogenous.
"""
const fcnone = FCNone()

"""
    const fcgiven = FCGiven()

Used when the final conditions are given in the exogenous data.
"""
const fcgiven = FCGiven()

"""
    const fclevel = FCMatchSSLevel()

Used when the final condition is to match the level of the steady state
solution.
"""
const fclevel = FCMatchSSLevel()

"""
    const fcslope = FCMatchSSRate()

Used when the final condition is to match the slope of the steady state
solution.
"""
const fcslope = FCMatchSSRate()

"""
    const fcrate = FCMatchSSRate()

Used when the final condition is to match the slope of the steady state
solution.
"""
const fcrate = FCMatchSSRate()

"""
    const fcnatural = FCConstRate()

Used when the final condition is a constant slope, but the slope value is
unknown.
"""
const fcnatural = FCConstRate()

export FinalCondition
export FCNone, fcnone
export FCGiven, fcgiven
export FCMatchSSLevel, fclevel
export FCMatchSSRate, fcslope, fcrate
export FCConstRate, fcnatural
export setfc, setfc!

"""
    setfc(model, fc)

Return a vector of final conditions for all variables in the `model`. The final
conditions of all variables are set to `fc`, except shocks and exogenous, which
are always `fcnone`. Use [`setfc!`](@ref) to update the final condition of
individual variable.
"""
function setfc end
function setfc(m::Model, fc::FinalCondition)
    ret = similar(m.allvars, FinalCondition)
    for (vi, v) in enumerate(m.allvars)
        ret[vi] = isshock(v) || isexog(v) ? fcnone : fc
    end
    return ret
end

"""
    setfc!(fc_vector, model, variable, new_fc)

Update the final condition for the given `variable` in the given `fc_vector` to
`new_fc`. The `fc_vector` is the output of [`setfc`](@ref).    
"""
function setfc! end
function setfc!(fc_vec::AbstractVector{FinalCondition}, m::Model, v::Union{ModelVariable,Symbol}, new_fc::FinalCondition)
    vi = ModelBaseEcon._index_of_var(v, m.allvars)
    if vi === nothing
        throw(ArgumentError("Unknown variable $(Symbol(v))."))
    end
    fc_vec[vi] = new_fc
    return fc_vec
end
