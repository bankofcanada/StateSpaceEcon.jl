##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################


"""
    SimData

Data structure containing the time series data for a simulation.

It is a collection of [`TSeries`](@ref) of the same frequency and containing
data for the same range. When used for simulation, the range must include the
initial conditions, the simulation range and the final conditions, although it
could extend beyond that. It must contain time series for all variables and
shocks in the model, in the same order as in the model object.

"""
const SimData = MVTSeries{F,Float64} where {F<:Frequency}
export SimData

# same constructors as should work for SimData
SimData(args...) = MVTSeries(args...)

_getname(::ModelVariable{NAME}) where {NAME} = NAME

const _MVCollection = Union{Vector{ModelVariable},NTuple{N,ModelVariable}} where {N}
# we should allow indexing with model variables
Base.getindex(sd::MVTSeries, vars::_MVCollection) = getindex(sd, _getname.(vars))
Base.getindex(sd::MVTSeries, vars::ModelVariable) = getindex(sd, _getname(vars))
Base.setindex!(sd::MVTSeries, val, vars::_MVCollection) = setindex!(sd, val, _getname.(vars))
Base.setindex!(sd::MVTSeries, val, vars::ModelVariable) = setindex!(sd, val, _getname(vars))

Base.getindex(sd::MVTSeries, rows, vars::_MVCollection) = getindex(sd, rows, _getname.(vars))
Base.getindex(sd::MVTSeries, rows, vars::ModelVariable) = getindex(sd, rows, _getname(vars))
Base.setindex!(sd::MVTSeries, val, rows, vars::_MVCollection) = setindex!(sd, val, rows, _getname.(vars))
Base.setindex!(sd::MVTSeries, val, rows, vars::ModelVariable) = setindex!(sd, val, rows, _getname(vars))

Base.view(sd::MVTSeries, vars::_MVCollection) = view(sd, :, _getname.(vars))
Base.view(sd::MVTSeries, vars::ModelVariable) = view(sd, :, _getname(vars))
Base.view(sd::MVTSeries, rows, vars::_MVCollection) = view(sd, rows, _getname.(vars))
Base.view(sd::MVTSeries, rows, vars::ModelVariable) = view(sd, rows, _getname(vars))

#######################################################

export array2data, array2workspace, data2array, data2workspace, workspace2array, workspace2data

"""
    array2data(matrix, model, plan; copy=false)
    array2data(matrix, vars, range; copy=false)

Convert a plain matrix with simulation data to a [`SimData`](@ref).
"""
function array2data end

"""
    array2workspace(matrix, model, plan; copy=false)
    array2workspace(matrix, vars, range; copy=false)

Convert a plain matrix with simulation data to a [`Workspace`](@ref
TimeSeriesEcon.Workspace).
"""
function array2workspace end


array2data(matrix::AbstractMatrix, model::Model, plan::Plan; copy=false) = array2data(matrix, model.varshks, plan.range; copy=copy)
array2data(matrix::AbstractMatrix, vars, range; copy=false) = SimData(range, vars, copy ? Base.copy(matrix) : matrix)

array2workspace(matrix::AbstractMatrix, model::Model, plan::Plan; copy=false) = array2workspace(matrix, model.varshks, plan.range; copy=copy)
array2workspace(matrix::AbstractMatrix, vars, range; copy=false) = Workspace(Symbol(v) => TSeries(range, copy ? Base.copy(matrix[:, i]) : matrix[:, i]) for (i, v) = enumerate(vars))

"""
    data2array(data; copy=false)
    data2array(data, model, plan; copy=false)
    data2array(data, vars, range; copy=false)

Convert a [`SimData`](@ref) to a matrix.
"""
function data2array end

data2array(simdata::SimData, m::Model, p::Plan; copy=false) = data2array(simdata, m.varshks, p.range; copy=copy)
data2array(simdata::SimData; copy=false) = copy ? Base.copy(rawdata(simdata)) : rawdata(simdata)
data2array(simdata::SimData, vars, range; copy=false) = copy ? Base.copy(rawdata(simdata[range, vars])) : rawdata(simdata[range, vars])

"""
    data2workspace(data; copy=false)
    data2workspace(data, model, plan; copy=false)
    data2workspace(data, vars, range; copy=false)
    
Convert a [`SimData`](@ref) to a [`Workspace`](@ref TimeSeriesEcon.Workspace).
"""
function data2workspace end

data2workspace(simdata::SimData, m::Model, p::Plan; copy=false) = data2workspace(simdata, m.varshks, p.range; copy=copy)
data2workspace(simdata::SimData; copy=false) = data2workspace(simdata, axes(simdata, 2), rangeof(simdata); copy=copy)
data2workspace(simdata::SimData, vars, range; copy=false) = Workspace(Symbol(v) => copy ? Base.copy(simdata[range, v]) : simdata[range, v] for v in vars)

"""
    workspace2array(w, model, plan; copy=false)
    workspace2array(w, vars, range; copy=false)

Convert a [`Workspace`](@ref TimeSeriesEcon.Workspace) to a matrix.
"""
function workspace2array end

workspace2array(w::Workspace, vars, range::AbstractUnitRange; copy=false) = hcat((w[Symbol(v)][range] for v in vars)...)
workspace2array(w::Workspace, model::Model, plan::Plan; copy=false) = workspace2array(w, model.varshks, plan.range; copy=copy)

"""
    workspace2data(w, model, plan; copy=false)
    workspace2data(w, vars, plan; copy=false)

Convert a [`SimData`](@ref) to a [`Workspace`](@ref TimeSeriesEcon.Workspace)
"""
function workspace2data end

function workspace2data(w::Workspace, vars, range::AbstractUnitRange; copy=false)
    ret = SimData(range, vars, NaN)
    for v in vars
        wv = w[Symbol(v)]
        copyto!(ret[v], intersect(range, rangeof(wv)), wv)
    end
    return ret
end
workspace2data(w::Workspace, model::Model, plan::Plan; copy=false) = workspace2data(w, model.varshks, plan.range; copy=copy)

workspace2data(w::Workspace, model::Model; copy=false) = workspace2data(w, model.varshks; copy=copy)
workspace2array(w::Workspace, model::Model; copy=false) = workspace2array(w, model.varshks; copy=copy)
function workspace2array(w::Workspace, vars; copy=true)
    range = mapreduce(v -> rangeof(w[Symbol(v)]), intersect, vars)
    return hcat((w[Symbol(v)][range] for v in vars)...)
end

@inline function workspace2data(w::Workspace, vars; copy=true)
    range = mapreduce(v -> rangeof(w[Symbol(v)]), intersect, vars)
    return workspace2data(w, vars, range; copy)
end

@inline function workspace2data(w::Workspace; copy=true)
    vars = collect(keys(w))
    return workspace2data(w, vars; copy)
end

"""
    dict2array, array2dict
    dict2data, data2dict

Deprecated. Use the workspace instead of dict.
"""
function dict2array end, function dict2data end, function array2dict end, function data2dict end

export dict2array, array2dict, dict2data, data2dict
@deprecate dict2array(d::AbstractDict, args...; kwargs...) workspace2array(Workspace(d), args...; kwargs...)
@deprecate dict2data(d::AbstractDict, args...; kwargs...) workspace2data(Workspace(d), args...; kwargs...)
@deprecate dict2array(d::Workspace, args...; kwargs...) workspace2array(d, args...; kwargs...)
@deprecate dict2data(d::Workspace, args...; kwargs...) workspace2data(d, args...; kwargs...)
@deprecate array2dict(args...; kwargs...) array2workspace(args...; kwargs...)
@deprecate data2dict(args...; kwargs...) data2workspace(args...; kwargs...)


struct SimFailed <: Exception
    info
end
Base.showerror(io::IO, ex::SimFailed) =
    isnothing(ex.info) ? print(io, "Simulation failed.") :
    ex.info isa MIT ? print(io, "Simulation failed in period $(ex.info).") :
    ex.info isa AbstractUnitRange{<:MIT} ? print(io, "Simulation over $(ex.info) failed.") :
    print(io, "Simulation failed: $(ex.info)")
isfailed(f::SimFailed)::Bool = !isnothing(f.info)
isfailed(f::SimData)::Bool = false
isfailed(f::Workspace)::Bool = false
isfailed(f)::Bool = throw(ArgumentError("Unexpected $(typeof(f)) argument."))
const MaybeSimData = Union{<:SimData,SimFailed}
Base.promote_rule(T::Type{<:SimData}, S::Type{<:SimFailed}) = Union{T, S}::Type
export SimFailed
export isfailed
export MaybeSimData

