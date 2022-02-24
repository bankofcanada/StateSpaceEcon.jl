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
    `TimeSeriesEcon.overlay` instead.  Note the important difference that
    in `TimeSeriesEcon.overlay` the values in the overlay come from the *first*
    series where the value exists, as opposed to `seriesoverlay` where it was
    from the last one.

See also: [`dictoverlay`](@ref)

"""
function seriesoverlay end
export seriesoverlay
@deprecate seriesoverlay(ts1::TSeries, ts2::TSeries) overlay(ts2, ts1)

#######################################################

"""
    dictoverlay(d1, d2)

Merge two dictionaries. Common key where the values are [`TSeries`](@ref) of the
same frequency are overlayed. Otherwise, a common key takes the value of the
last Dict containing it.

!!! note "Deprecation Note"

    This function will be removed. Use `TimeSeriesEcon.overlay` instead. An
    important difference is that `TimeSeriesEcon.overlay` keeps the values from
    the first argument where the key appears, while `dictoverlay` keeps it from
    the last.

"""
function dictoverlay end
export dictoverlay
@deprecate dictoverlay(D1::Dict{String,<:Any}, D2::Dict{String,<:Any}) overlay(Workspace(D2), Workspace(D1))

#######################################################

"""
    array2data(matrix, vars, range)
    array2data(matrix, model, plan)
    array2workspace(matrix, vars, range)
    array2workspace(matrix, model, plan)
    data2array(simdata)
    data2workspace(simdata)
    workspace2array(w, vars, range)
    workspace2array(w, model, plan)
    workspace2data(w, vars, range)
    workspace2data(w, model, plan)
"""
(
    function array2data end,
    function array2workspace end,
    function data2array end,
    function data2workspace end,
    function workspace2array end,
    function workspace2data end
)
export array2data, array2workspace, data2array, data2workspace, workspace2array, workspace2data

@inline array2data(matrix::AbstractMatrix, model::Model, plan::Plan; copy = false) = array2data(matrix, model.varshks, plan.range; copy = copy)
@inline array2data(matrix::AbstractMatrix, vars, range; copy = false) = SimData(range, vars, copy ? Base.copy(matrix) : matrix)

@inline array2workspace(matrix::AbstractMatrix, model::Model, plan::Plan; copy = false) = array2workspace(matrix, model.varshks, plan.range; copy = copy)
@inline array2workspace(matrix::AbstractMatrix, vars, range; copy = false) = Workspace(Symbol(v) => TSeries(range, copy ? Base.copy(matrix[:, i]) : matrix[:, i]) for (i, v) = enumerate(vars))

@inline data2array(simdata::SimData, m::Model, p::Plan; copy = false) = data2array(simdata, m.varshks, p.range; copy = copy)
@inline data2array(simdata::SimData; copy = false) = copy ? Base.copy(rawdata(simdata)) : rawdata(simdata)
@inline data2array(simdata::SimData, vars, range; copy = false) = copy ? Base.copy(rawdata(simdata[range, vars])) : rawdata(simdata[range, vars])
@inline data2workspace(simdata::SimData, m::Model, p::Plan; copy = false) = data2workspace(simdata, m.varshks, p.range; copy = copy)
@inline data2workspace(simdata::SimData; copy = false) = data2workspace(simdata, axes(simdata,2), rangeof(simdata); copy = copy)
@inline data2workspace(simdata::SimData, vars, range; copy = false) = Workspace(Symbol(v) => copy ? Base.copy(simdata[range, v]) : simdata[range, v] for v in vars)

@inline workspace2array(w::Workspace, vars, range::AbstractUnitRange; copy = false) = hcat((w[Symbol(v)][range] for v in vars)...)
@inline workspace2array(w::Workspace, model::Model, plan::Plan; copy = false) = workspace2array(w, model.varshks, plan.range; copy = copy)

function workspace2data(w::Workspace, vars, range::AbstractUnitRange; copy = false)
    ret = SimData(range, vars)
    for v in vars
        copyto!(ret[v], range, w[Symbol(v)])
    end
    return ret
end
@inline workspace2data(w::Workspace, model::Model, plan::Plan; copy = false) = workspace2data(w, model.varshks, plan.range; copy = copy)

@inline workspace2data(w::Workspace, model::Model; copy = false) = workspace2data(w, model.varshks; copy = copy)
@inline workspace2array(w::Workspace, model::Model; copy = false) = workspace2array(w, model.varshks; copy = copy)
function workspace2array(w::Workspace, vars; copy = false)
    range = mapreduce(v -> rangeof(w[Symbol(v)]), intersect, vars)
    return hcat((w[Symbol(v)][range] for v in vars)...)
end

function workspace2data(w::Workspace, vars; copy = false)
    range = mapreduce(v -> rangeof(w[Symbol(v)]), intersect, vars)
    return hcat(MVTSeries(range); (v => w[Symbol(v)] for v in vars)...)
end

export dict2array, array2dict, dict2data, data2dict
@deprecate dict2array(d::AbstractDict, args...; kwargs...) workspace2array(Workspace(d), args...; kwargs...)
@deprecate dict2data(d::AbstractDict, args...; kwargs...) workspace2data(Workspace(d), args...; kwargs...)
@deprecate dict2array(d::Workspace, args...; kwargs...) workspace2array(d, args...; kwargs...)
@deprecate dict2data(d::Workspace, args...; kwargs...) workspace2data(d, args...; kwargs...)
@deprecate array2dict(args...; kwargs...) array2workspace(args...; kwargs...)
@deprecate data2dict(args...; kwargs...) data2workspace(args...; kwargs...)
