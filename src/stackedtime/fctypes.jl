##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

abstract type FinalCondition end
abstract type NoFinalCondition <: FinalCondition end
struct FCNone <: NoFinalCondition end
abstract type HasFinalCondition <: FinalCondition end
struct FCGiven <: HasFinalCondition end
struct FCMatchSSLevel <: HasFinalCondition end
struct FCMatchSSRate <: HasFinalCondition end
struct FCConstRate <: HasFinalCondition end

const fcnone = FCNone()
const fcgiven = FCGiven()
const fclevel = FCMatchSSLevel()
const fcslope = FCMatchSSRate()
const fcrate = FCMatchSSRate()
const fcnatural = FCConstRate()

export FinalCondition
export NoFinalCondition
export HasFinalCondition
export FCNone, fcnone
export FCGiven, fcgiven
export FCMatchSSLevel, fclevel
export FCMatchSSRate, fcslope, fcrate
export FCConstRate, fcnatural
export setfc

function setfc(m::Model, fc::FinalCondition)
    ret = similar(m.allvars, FinalCondition)
    for (vi, v) in enumerate(m.allvars)
        ret[vi] = isshock(v) || isexog(v) ? fcnone : fc
    end
    return ret
end

function setfc!(fc_vec::AbstractVector{FinalCondition}, m::Model, v::ModelVariable, new_fc::FinalCondition)
    @assert m.allvars[v.index] == v
    fc_vec[v.index] = new_fc
    return fc_vec
end

function setfc!(fc_vec::AbstractVector{FinalCondition}, m::Model, v::Symbol, new_fc::FinalCondition)
    vi = indexin([v], m.allvars)[1]
    if vi === nothing
        throw(ArgumentError("Unknown variable $(Symbol(v))."))
    end
    @assert vi == m.allvars[vi].index
    fc_vec[vi] = new_fc
    return fc_vec
end


