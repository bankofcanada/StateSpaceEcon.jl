##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

import ..SimData
import ..rawdata

"""
    seriesoverlay(ts1, ts2)

Return a new [`TSeries`](@ref) over the full range of both arguments. The overlapping part
contains values from the last argument.

!!! note "Deprecation Note"

    This function will be removed in the future. Use
    [`TimeSeriesEcon.overlay`](@ref) instead. Note the important difference that
    in [`TimeSeriesEcon.overlay`](@ref) the values in the overlay come from the
    *first* series where the value exists, as opposed to `seriesoverlay` where
    it was from the last one.

See also: [`dictoverlay`](@ref)

"""
function seriesoverlay end
export seriesoverlay
@deprecate seriesoverlay(ts1::TSeries, ts2::TSeries) overlay(ts2, ts1)

#######################################################

"""
    dictoverlay(d1, d2)

Merge two dictionaries. Common key where the values are [`TSeries`](@ref
TimeSeriesEcon.TSeries) of the same frequency are overlayed. Otherwise, a common
key takes the value of the last Dict containing it.

!!! note "Deprecation Note"

    This function will be removed. Use [`TimeSeriesEcon.overlay`](@ref) instead.
    An important difference is that [`TimeSeriesEcon.overlay`](@ref) keeps the values
    from the first argument where the key appears, while `dictoverlay` keeps it
    from the last.

"""
function dictoverlay end
export dictoverlay
@deprecate dictoverlay(D1::Dict{String,<:Any}, D2::Dict{String,<:Any}) overlay(Workspace(D2), Workspace(D1))

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
    ret = SimData(range, vars)
    for v in vars
        copyto!(ret[v], range, w[Symbol(v)])
    end
    return ret
end
workspace2data(w::Workspace, model::Model, plan::Plan; copy=false) = workspace2data(w, model.varshks, plan.range; copy=copy)

workspace2data(w::Workspace, model::Model; copy=false) = workspace2data(w, model.varshks; copy=copy)
workspace2array(w::Workspace, model::Model; copy=false) = workspace2array(w, model.varshks; copy=copy)
function workspace2array(w::Workspace, vars; copy=false)
    range = mapreduce(v -> rangeof(w[Symbol(v)]), intersect, vars)
    return hcat((w[Symbol(v)][range] for v in vars)...)
end

function workspace2data(w::Workspace, vars; copy=false)
    range = mapreduce(v -> rangeof(w[Symbol(v)]), intersect, vars)
    return hcat(MVTSeries(range); (v.name => w[Symbol(v)] for v in vars)...)
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
