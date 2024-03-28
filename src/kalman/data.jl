##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

# this file declares the data structures used by the
# Kalman filter algorithms in this module


"""
    AbstractKFData{RANGE,NS,NO,T}

The parent class for all Kalman filter data types. 

`RANGE` - a positive integer, or a range. Determines the number of periods and
the range of values of `t` that will be passed in calls to
[`kf_predict_x!`](@ref), [`kf_predict_y!`](@ref), and [`kf_true_y!`](@ref).

`NS` - number of state variables, as returned by [`kf_length_x`](@ref)

`NO` - number of observed variables, as returned by [`kf_length_y`](@ref)

`T<:Real` - type of values, typically `Float64`.

Concrete types derived from `AbstractKFData` may have some or all of the
following fields:
* `x_pred` - predicted state, i.e. E(xₜ | y₁, ..., yₜ₋₁)
* `Px_pred` - covariance of the predicted state
* `x` - corrected/filtered state, i.e. E[xₜ | y₁, ..., yₜ]
* `Px` - covariance of corrected/filtered state
* `y` - true observations, as returned by `kf_true_y!`
* `y_pred` - filtered observation, i.e. E[yₜ | y₁, ..., yₜ₋₁]
* `Py_pred` - covariance of filtered observation
* `error_y` - observation error ( y - y_pred)
* `Pxy_pred` - cross-covariance of x_pred and y_pred
* `K` - Kalman gain matrix
* `x_smooth` - smoothed state, i.e. E[xₜ | all y]
* `Px_smooth` - covariance of smoothed state
* `y_smooth` - smoothed observation, i.e. E[yₜ | all y]
* `Py_smooth` - covariance of smoothed observation
* `J` - Kalman smoother matrix
* `loglik` - log likelihood based on Kalman filter 
* `res2` - sum of squared observation residuals (error_y' * error_y)

"""
abstract type AbstractKFData{RANGE,NS,NO,T} end

Base.eltype(::Type{<:AbstractKFData{RANGE,NS,NO,T}}) where {RANGE,NS,NO,T} =
    @isdefined(T) ? T : Real


function TimeSeriesEcon.compare_equal(x::T, y::T; kwargs...) where {T<:AbstractKFData}
    equal = true
    for prop in propertynames(x)
        if !compare(getproperty(x, prop), getproperty(y, prop), prop; kwargs...)
            equal = false
        end
    end
    return equal
end

function TimeSeriesEcon.compare_equal(x::T1, y::T2; kwargs...) where {T1<:AbstractKFData, T2<:AbstractKFData}
    equal = true
    for prop in union(propertynames(x), propertynames(y))
        propx = hasproperty(x, prop) ? getproperty(x, prop) : missing
        propy = hasproperty(y, prop) ? getproperty(y, prop) : missing
        if !compare(propx, propy, prop; kwargs...)
            equal = false
        end
    end
    return equal
end

"""
    struct _KFValueInfo ... end

Internal struct - holds information about a given filed, such as dimensions and
description.
"""
struct _KFValueInfo
    dims::Tuple
    description::String
end

const KFDataInfo = (;
    x_pred=_KFValueInfo((:NS,), "predicted state, i.e. E(xₜ | y₁, ..., yₜ₋₁)"),
    Px_pred=_KFValueInfo((:NS, :NS), "covariance of the predicted state"),
    x=_KFValueInfo((:NS,), "corrected/filtered state, i.e. E[xₜ | y₁, ..., yₜ]"),
    Px=_KFValueInfo((:NS, :NS), "covariance of corrected/filtered state"),
    y=_KFValueInfo((:NO,), "true observations, as returned by `kf_true_y!`"),
    y_pred=_KFValueInfo((:NO,), "filtered observation, i.e. E[yₜ | y₁, ..., yₜ₋₁]"),
    Py_pred=_KFValueInfo((:NO, :NO), "covariance of filtered observation"),
    Pxy_pred=_KFValueInfo((:NS, :NO), "cross-covariance of x_pred and y_pred"),
    error_y=_KFValueInfo((:NO,), "observation error ( y - y_pred)"),
    K=_KFValueInfo((:NS, :NO), "Kalman gain matrix"),
    x_smooth=_KFValueInfo((:NS,), "smoothed state, i.e. E[xₜ | all y]"),
    Px_smooth=_KFValueInfo((:NS, :NS), "covariance of smoothed state"),
    Pxx_smooth=_KFValueInfo((:NS, :NS), "cross-covariance of xₜ and xₜ₊₁"),
    y_smooth=_KFValueInfo((:NO,), "smoothed observation, i.e. E[yₜ | all y]"),
    Py_smooth=_KFValueInfo((:NO, :NO), "covariance of smoothed observation"),
    J=_KFValueInfo((:NS, :NS), "Kalman smoother matrix"),
    loglik=_KFValueInfo((), "log likelihood based on Kalman filter "),
    res2=_KFValueInfo((), "sum of squared observation residuals (error_y' * error_y)"),
    Ly_pred=_KFValueInfo((:NO, :NO), "Cholesky lower triangular factor of Py_pred"),
    aux_ZᵀPy⁻¹=_KFValueInfo((:NS, :NO), "Auxiliary matrix Zᵀ⋅Py⁻¹")
)

_offset_expr(::Integer) = :t
_offset_expr(RANGE::UnitRange) = (offset = first(RANGE) - 1; :(Int(t - $offset)))

Base.view(kfd::AbstractKFData, t::Integer, name::Symbol) = kfd_getvalue(view, kfd, t, Val(name))
Base.view(kfd::AbstractKFData, t::Integer, name::Val) = kfd_getvalue(view, kfd, t, name)
Base.getindex(kfd::AbstractKFData, t::Integer, name::Symbol) = kfd_getvalue(getindex, kfd, t, Val(name))
Base.getindex(kfd::AbstractKFData, t::Integer, v::Val) = kfd_getvalue(getindex, kfd, t, v)
@generated function kfd_getvalue(access::Function, kfd::AbstractKFData{RANGE}, t::Integer, ::Val{NAME}) where {RANGE,NAME}
    vinfo = get(KFDataInfo, NAME, nothing)
    # if a non-standard field is requested, we return an error
    if isnothing(vinfo)
        return :(error("type $(kfd) has no field $NAME"))
    end
    # if one of the standard fields is requested but missing, we return `nothing`
    if !hasfield(kfd, NAME)
        return nothing
    end
    tt = _offset_expr(RANGE)
    # figure out the number of dimensions in this field
    ndims = length(vinfo.dims)
    if ndims == 0
        return :(access(kfd.$NAME, $tt))
    elseif ndims == 1
        return :(access(kfd.$NAME, :, $tt))
    elseif ndims == 2
        return :(access(kfd.$NAME, :, :, $tt))
    else
        return :(error("Unexpected value with ndims = $ndims"))
    end
end


# function Base.setindex!(kfd::AbstractKFData, values::Tuple, t::Integer, names::Symbol...)
#     if length(value) != length(names)
#         throw(ArgumentError("number of values on the right doesn't match number of names within the brackets."))
#     end
#     for (val, name) in zip(values, names)
#         kf_setvalue!(kfd, val, t, Val(name))
#     end
#     values
# end
Base.setindex!(kfd::AbstractKFData, value, t::Integer, name::Symbol) = kfd_setvalue!(kfd, value, t, Val(name))
@generated function kfd_setvalue!(kfd::AbstractKFData{RANGE}, value, t::Integer, ::Val{NAME}) where {RANGE,NAME}
    vinfo = get(KFDataInfo, NAME, nothing)
    # if a non-standard field is requested, we return an error
    if isnothing(vinfo)
        return :(error("type $(kfd) has no field $NAME"))
    end
    # if one of the standard fields is requested but missing, do nothing
    if !hasfield(kfd, NAME)
        return nothing
    end
    tt = _offset_expr(RANGE)
    ndims = length(vinfo.dims)
    if ndims == 0
        return :(kfd.$NAME[$tt] = value)
    elseif ndims == 1
        # return :(copyto!(view(kfd.$NAME, :, $tt), value))
        return :(kfd.$NAME[:, $tt] = value)
    elseif ndims == 2
        # return :(copyto!(view(kfd.$NAME, :, :, $tt), value))
        return :(kfd.$NAME[:, :, $tt] = value)
    else
        return :(error("Unexpected value with ndims = $ndims"))
    end
end

#############################################################################
# the following macro can be used to create a custom data struct

function _new_kf_data(expr)
    if !@capture(expr, struct Name_
        FIELDS__
    end)
        error("Argument must be a struct Name ... end")
    end
    Name isa Symbol || error("Argument must be a simple struct without type parameters or inheritance specifications.")
    if !all(x -> x isa Symbol, FIELDS)
        error("Argument must be a simple struct where only the field names are given and without inner constructors.")
    end
    alloc_exprs = []
    struct_expr = MacroTools.postwalk(expr) do x
        x isa Bool && return x
        x isa LineNumberNode && return x
        x === Name && return :($Name{RANGE,NS,NO,T} <: $(@__MODULE__).AbstractKFData{RANGE,NS,NO,T})
        x isa Symbol && begin
            haskey(KFDataInfo, x) || error("$x is not a valid field name for Kalman filter data")
            xdims = KFDataInfo[x].dims
            ndims = 1 + length(xdims)
            push!(alloc_exprs, :(Array{T}(undef, $(xdims...), nperiods)))
            return :($x::Array{T,$ndims})
        end
        x isa Expr && return x
        error("Can't process x of type $(typeof(x))")
    end
    isempty(alloc_exprs) && error("Must specify at least one field")
    constr_expr = MacroTools.striplines(:(
        function $Name(T::Type{<:Real}, rng::Union{Integer,UnitRange{<:Integer}}, model, user_data...)
            NS = kf_length_x(model, user_data...)
            NO = kf_length_y(model, user_data...)
            nperiods = rng isa UnitRange ? length(rng) : rng
            return $Name{rng,NS,NO,T}($(alloc_exprs...))
        end
    ))
    constr2_expr = MacroTools.striplines(:(
        $Name(rng::Union{Integer,UnitRange{<:Integer}}, model, user_data...) = $Name(Float64, rng, model, user_data...)
    ))
    return Expr(:block, struct_expr, constr_expr, constr2_expr)
end

macro kf_data_struct(expr)
    return esc(_new_kf_data(expr))
end

"""
    struct KFDataFilter <: AbstractKFData ... end

A structure type that contains the most used outputs of the Kalman filter.
Namely, `x_pred`, `Px_pred`, `y_pred`, `Py_pred`, `x`, `Px`, `loglik`.

Related [`KFDataFilterEx`](@ref) contains all outputs.
"""
KFDataFilter
@kf_data_struct struct KFDataFilter
    x_pred
    Px_pred
    y_pred
    Py_pred
    x
    Px
    loglik
end

"""
    struct KFDataFilterEx <: AbstractKFData ... end

A structure type that contains the all outputs of the Kalman filter.
Namely, `x_pred`, `x`, `Px_pred`, `Px`, `y_pred`, `Py_pred`, `loglik`.
"""
KFDataFilterEx
@kf_data_struct struct KFDataFilterEx
    x_pred
    Px_pred
    y_pred
    Py_pred
    Pxy_pred
    x
    Px
    K
    error_y
    res2
    loglik
end

@kf_data_struct struct KFDataSmoother
    x_pred
    Px_pred
    y_pred
    Py_pred
    # Ly_pred
    aux_ZᵀPy⁻¹
    K
    error_y
    x_smooth
    Px_smooth
    y_smooth
    Py_smooth
    Pxx_smooth
    loglik
end

@kf_data_struct struct KFDataSmootherEx
    x_pred
    Px_pred
    y_pred
    Py_pred
    # Ly_pred
    aux_ZᵀPy⁻¹
    Pxy_pred
    x
    Px
    K
    error_y
    x_smooth
    Px_smooth
    y_smooth
    Py_smooth
    Pxx_smooth
    J
    res2
    loglik
end

macro kfd_set!(kfd, t, values::Symbol...)
    ret = MacroTools.striplines(quote end)
    push!(ret.args, __source__)
    for val in values
        push!(ret.args,
            :(kfd_setvalue!($kfd, $val, $t, Val($(Meta.quot(val)))))
        )
    end
    return esc(ret)
end

macro kfd_get(kfd, t, value::Symbol)
    return esc(:(
        $kfd[$t, Val($(Meta.quot(value)))]
    ))
end

macro kfd_view(kfd, t, value::Symbol)
    return esc(:(
        view($kfd, $t, Val($(Meta.quot(value))))
    ))
end

