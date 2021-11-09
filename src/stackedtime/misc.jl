##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

import ..SimData
import ..rawdata

export seriesoverlay
"""
    seriesoverlay(ts1, ts2)

Return a new [`TSeries`](@ref) over the full range of both arguments. The overlapping part
contains values from the last argument.

See also: [`dictoverlay`](@ref)

"""
function seriesoverlay(ts1::TSeries, ts2::TSeries)
    # Make a copy of the output sries
    tsout = deepcopy(ts2);
    # Range of the first TSeries
    Rng1 = mitrange(ts1);
    # Range of the second TSeries
    Rng2 = mitrange(ts2);
    if Rng1.start < Rng2.start
        tsout[Rng1.start:Rng2.start - 1] = ts1;
    end
    if Rng1.stop > Rng2.stop
        tsout[Rng2.stop + 1:Rng1.stop] = ts1;
    end
    return tsout
end

export dictoverlay
"""
    dictoverlay(d1, d2)

Merge two dictionaries. Common key where the values are [`TSeries`](@ref) of the
same frequency are overlayed. Otherwise, a common key takes the value of the
last Dict containing it.

See also: [`seriesoverlay`](@ref)
"""
function dictoverlay(D1::Dict{String,<:Any}, D2::Dict{String,<:Any})
    # Get keys
    K1 = keys(D1);
    K2 = keys(D2);
    # Combine the keys
    K = unique([collect(K1) ; collect(K2)]);
    # Pre-allocate output
    D3 = Dict{String,Any}();
    # Go over each key and update D3
    for k in K
        # Test where we can find the key
        t1 = in(k, K1)
        t2 = in(k, K2)
        if !t1
            # If we cannot find it in K1, we use D2
            push!(D3, k => D2[k])
        elseif !t2
            # If we cannot find it in K2, we use D1
            push!(D3, k => D1[k])
        else
            # If we find it in both D1 and D2
            if isa(D1[k], TSeries) && isa(D2[k], TSeries)
                # If both are TSeries, we overlay them.
                push!(D3, k => seriesoverlay(D1[k], D2[k]))
            else
                # Otherwise, we give priority to D2
                push!(D3, k => D2[k])
            end
        end
    end
    return D3 = deepcopy(D3)
end

export array2dict
"""
    array2dict(data, vars, start_date)

Convert the simulation data array to a dictionary.
"""
function array2dict(data::AbstractArray{Float64,2}, vars::AbstractVector, start_date::MIT)::Dict{String,Any}
    Dict{String,Any}(string(vars[i]) => TSeries(start_date, data[:,i]) for i ∈ 1:length(vars))
end

export array2data
"""
    array2data(data, vars, start_date; copy=false)

Convert the simulation data array to a [`SimData`](@ref).

Use the `copy` argument to control whether or not the returned SimData will hold
a reference to the given data array or its own copy.

"""
function array2data(data::AbstractArray{Float64,2}, vars::AbstractVector, start_date::MIT; copy=false)::SimData
    SimData(start_date, vars, ifelse(copy, Base.copy(data), data))
end

export dict2array

@inline _to_string(a::AbstractString) = a
@inline _to_string(a::Symbol) = string(a)
@inline _to_string(a::ModelVariable) = string(a.name)

function _d2_vars(d, vars)
    if keytype(d) <: AbstractString
        vars = _to_string.(vars)
    else
        vars = keytype(d).(vars)
    end
    missing_vars = setdiff(vars, keys(d))
    if !isempty(missing_vars)
        throw(ArgumentError("""Variables not found in data dictionary: $(join(missing_vars, ", "))"""))
    end
    return vars
end

function _d2_rng(d, vars, range)
    ranges = [mitrange(d[v]) for v in vars]
    fs = first.(ranges)
    ls = last.(ranges)
    if isempty(range)
        f = maximum(fs)
        l = minimum(ls)
        range = f:l
        if !(all(f .== fs) && all(l .== ls))
            @warn "Variable ranges are not all the same. Specify `range=` to avoid this warning. Using intersection range: $(range)"
        end
    else
        missing_data = (first(range) .< fs) .| (last(range) .> ls)
        if any(missing_data)
            throw(ArgumentError("""Data is not available for the full range $(range) for some variables: $(join(vars[missing_data], ", "))"""))
        end
    end
    return range
end

"""
    dict2array(d, vars; range)

Convert a dictionary of [`TSeries`](@ref) to a 2d array of simulation data for
the given range. If the `range` argument is not provided, the effective range is
the intersection of the ranges of available data for the given list of
variables.

"""
function dict2array(d::Dict, vars::AbstractVector; range::AbstractUnitRange=1:0)::Array{Float64,2}
    vars = _d2_vars(d, vars)
    range = _d2_rng(d, vars, range)
    data = Array{Float64,2}(undef, length(range), length(vars))
    for i in eachindex(vars)
        data[:, i] .= d[vars[i]][range].values
    end
    return data
end

export dict2data
"""
    dict2data(d, vars; range)

Convert a dictionary of [`TSeries`](@ref) to a `SimData`` for the given range.
If the `range` argument is not provided, the effective range is the intersection
of the ranges of available data for the given list of variables.

"""
function dict2data(d::Dict, vars::AbstractVector; range::AbstractUnitRange=1:0)::SimData
    vars = _d2_vars(d, vars)
    range = _d2_rng(d, vars, range)
    data = Array{Float64,2}(undef, length(range), length(vars))
    for i in eachindex(vars)
        data[:, i] .= d[vars[i]][range].values
    end
    return SimData(first(range), vars, data)
end

export data2dict
"""
    data2dict(sd::SimData; copy=false)::Dict{String, Any}

Convert a [`SimData`](@ref) to a dictionary containing the same data as
individual [`TSeries`](@ref).

Use the `copy` argument to control whether the TSeries in the returned Dict will
hold references to the columns of `sd` or copies of that data.

"""
function data2dict(sd::SimData; copy=false)
    transform = ifelse(copy, Base.copy, Base.identity)
    Dict{String,Any}(string(n) => transform(v) for (n, v) in pairs(sd))
end

export data2array
"""
    data2array(sd::SimData; copy=false)::Array{Float64,2}

Convert a [`SimData`](@ref) to an Array. The `copy` argument controls whether
the returned Array holds a copy of the data or a reference to the data in `sd`.

"""
function data2array(sd::SimData; copy=false)
    return ifelse(copy, Base.copy(rawdata(sd)), rawdata(sd))
end
