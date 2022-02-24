##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


"""
    SimData

Data structure containing the time series data for a simulation.

It is a collection of [`TSeries`](@ref) of the same frequency and containing
data for the same range. When used for simulation, the range must include the
initial conditions, the simulation range and the final conditions, although it
could extend beyond that. It must contain time series for all variables and
shocks in the model, although it might contain other time series.

"""
const SimData = MVTSeries{F,Float64} where {F<:Frequency}
export SimData

# same constructors as should work for SimData
SimData(args...) = MVTSeries(args...)

const _MVCollection = Union{Vector{ModelVariable},NTuple{N,ModelVariable}} where {N}
# we should allow indexing with model variables
Base.getindex(sd::SimData, vars::_MVCollection) = getindex(sd, map(v -> v.name, vars))
Base.getindex(sd::SimData, vars::ModelVariable) = getindex(sd, vars.name)
Base.setindex!(sd::SimData, val, vars::_MVCollection) = setindex!(sd, val, map(v -> v.name, vars))
Base.setindex!(sd::SimData, val, vars::ModelVariable) = setindex!(sd, val, vars.name)

Base.getindex(sd::SimData, rows, vars::_MVCollection) = getindex(sd, rows, map(v->v.name, vars))
Base.getindex(sd::SimData, rows, vars::ModelVariable) = getindex(sd, rows, vars.name)
Base.setindex!(sd::SimData, val, rows, vars::_MVCollection) = setindex!(sd, val, rows, map(v->v.name, vars))
Base.setindex!(sd::SimData, val, rows, vars::ModelVariable) = setindex!(sd, val, rows, vars.name)

Base.view(sd::SimData, rows, vars::_MVCollection) = view(sd, rows, map(v->v.name, vars))
Base.view(sd::SimData, rows, vars::ModelVariable) = view(sd, rows, vars.name)
