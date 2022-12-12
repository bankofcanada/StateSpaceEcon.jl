##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

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

