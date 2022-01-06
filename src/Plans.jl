##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    Plans

Module part of StateSpaceEcon. This module implements the `Plan` data structure,
which is used in simulations. The plan object contains information about the
range of the simulation and which variables and shocks are exogenous or
endogenous at each period of the range.

### Constructors
  * [`Plan(model, range)`](@ref Plan)

### Modify the plan
  * [`exogenize!`](@ref), [`endogenize!`](@ref) - make variables exogenous or
    endogenous
  * [`exog_endo!`](@ref), [`endo_exog!`](@ref) - swap exogenous and endogenous
    variables
  * [`autoexogenize!`](@ref) - exogenize and endogenize variables according to
    the list in the model

### Prepare data for simulation
  * [`zeroarray`](@ref), [`zerodict`](@ref), [`zerodata`](@ref) - prepare a
    matrix or a dictionary or a [`SimData`](@ref) of data for the simulation
    containing zeros.
  * [`steadystatearray`](@ref), [`steadystatedict`](@ref),
    [`steadystatedata`](@ref) - prepare a matrix or a dictionary or a [`SimData`](@ref) of data for the
    simulation containing the steady state.

"""
module Plans

using TimeSeriesEcon
using ModelBaseEcon

# exported from this module that are also exported from StateSpaceEcon
export Plan,
    exogenize!, endogenize!, exog_endo!, endo_exog!, autoexogenize!,
    zeroarray, zerodict, steadystatearray, steadystatedict, zerodata, steadystatedata

# exported from this module, but meant for internal use in StateSpaceEcon
export plansum


"""
    Plan{T <: MIT}

A data structure representing the simulation plan. It holds information about
the time range of the simulation and which variables/shocks are exogenous at
each period.

"""
struct Plan{T <: MIT} <: AbstractVector{Vector{Symbol}}
    range::AbstractUnitRange{T}
    varshks::NamedTuple
    exogenous::BitArray{2}
end

Plan(model::Model, r::Union{MIT,Int}) = Plan(model, r:r)

"""
    Plan(model, range)

Create a default simulation plan for the given model over the given range. The
range of the plan is augmented to include periods before and after the given
range, over which initial and final conditions will be applied. 

Instead of a range, one could also pass in a single moment in time
([`MIT`](@ref TimeSeriesEcon.MIT)) instance, in which case it is interpreted as
a range of length 1.

"""
function Plan(model::Model, range::AbstractUnitRange)
    if !(eltype(range) <: MIT)
        range = UnitRange{MIT{Unit}}(range)
    end
    range = (first(range) - model.maxlag):(last(range) + model.maxlead)
    local varshks = model.varshks
    local N = length(varshks)
    local names = tuple(Symbol.(varshks)...) # force conversion of Vector{ModelSymbol} to NTuple{N,Symbol}
    p = Plan(range, NamedTuple{names}(1:N), falses(length(range), length(varshks)))
    for (ind, var) = enumerate(varshks)
        if isshock(var) || isexog(var)
            p.exogenous[:, ind] .= [true]
        end
    end
    return p
end

#######################################
# AbstractVector interface 

Base.size(p::Plan) = size(p.range)
Base.axes(p::Plan) = (p.range,)
Base.length(p::Plan) = length(p.range)
Base.IndexStyle(::Plan) = IndexLinear()
Base.similar(p::Plan) = Plan(p.range, p.varshks, similar(p.exogenous))
Base.copy(p::Plan) = Plan(p.range, p.varshks, copy(p.exogenous))

@inline _offset(p::Plan{T}, idx::T) where {T <: MIT} = convert(Int, idx - first(p.range) + 1)
@inline _offset(p::Plan{T}, idx::AbstractUnitRange{T}) where {T <: MIT} = 
    _offset(p, first(idx)):_offset(p, last(idx))

## Integer index is taken as is
Base.getindex(p::Plan, idx::Int) = p[p.range[idx]]
Base.getindex(p::Plan, idx::AbstractUnitRange{Int}) = p[p.range[idx]]
Base.getindex(p::Plan, idx::Int, ::Val{:inds}) = findall(p.exogenous[idx,:])

# Index of the frequency type returns the list of exogenous symbols
Base.getindex(p::Plan{T}, idx::T) where T <: MIT = [keys(p.varshks)[p[_offset(p, idx),Val(:inds)]]...,]
function Base.getindex(p::Plan{T}, idx::T, ::Val{:inds}) where T <: MIT
    first(p.range) ≤ idx ≤ last(p.range) || throw(BoundsError(p, idx))
    return p[_offset(p, idx), Val(true)]
end

# A range returns the plan trimmed over that exact range.
Base.getindex(p::Plan{MIT{Unit}}, rng::AbstractUnitRange{Int}) = p[UnitRange{MIT{Unit}}(rng)]
function Base.getindex(p::Plan{T}, rng::AbstractUnitRange{T}) where T <: MIT
    rng.start < p.range.start && throw(BoundsError(p, rng.start))
    rng.stop > p.range.stop && throw(BoundsError(p, rng.stop))
    return Plan{T}(rng, p.varshks, p.exogenous[_offset(p, rng), :])
end

# A range with a model returns a plan trimmed over that range and extended for initial and final conditions.
Base.getindex(p::Plan{MIT{Unit}}, rng::AbstractUnitRange{Int}, m::Model) = p[UnitRange{MIT{Unit}}(rng), m]
function Base.getindex(p::Plan{T}, rng::AbstractUnitRange{T}, m::Model) where T <: MIT
    rng = (rng.start - m.maxlag):(rng.stop + m.maxlead)
    return p[rng]
end

Base.setindex!(p::Plan, x, i...) = error("Cannot assign directly. Use `exogenize` and `endogenize` to alter plan.")

#######################################
# Pretty printing

Base.summary(io::IO, p::Plan) = print(io, typeof(p), " with range ", p.range)

# Used in the show() implementation below
function collapsed_range(p::Plan{T}) where T <: MIT
    ret = Pair{Union{T,UnitRange{T}},Vector{Symbol}}[]
    i1 = first(p.range)
    i2 = i1
    val = p[i1]
    make_key() = i1 == i2 ? i1 : i1:i2
    for i in p.range[2:end]
        val_i = p[i]
        if val_i == val
            i2 = i
        else
            push!(ret, make_key() => val)
            i1 = i2 = i
            val = val_i
        end
    end
    push!(ret, make_key() => val)
end

Base.show(io::IO, ::MIME"text/plain", p::Plan) = show(io, p)
function Base.show(io::IO, p::Plan)
    # 0) show summary before setting :compact
    summary(io, p)
    isempty(p) && return
    # print(io, ":")
    nrow, ncol = displaysize(io)
    limit = get(io, :limit, true)
    cp = collapsed_range(p)
    # find the longest string left of "=>" for padding
    maxl = maximum(length("$k") for (k, v) in cp)
    if limit
        dcol = ncol - maxl - 6
    else
        dcol = typemax(Int)
    end
    function print_exog(names) 
        if isempty(names)
            print(io, "∅")
            return
        end
        lens = cumsum(map(x -> length("$x, "), names))
        show = lens .< dcol
        show[1] = true
        print(io, join(names[show], ", "))
        if !all(show)
            print(io, ", …")
        end
    end
    if !limit || length(cp) <= nrow - 5 
        for (r, v) in cp
            print(io, "\n  ", lpad("$r", maxl, " "), " → ")
            print_exog(v)
        end
    else
        top = div(nrow - 5, 2)
        bot = length(cp) - nrow + 6 + top
        for (r, v) in cp[1:top]
            print(io, "\n  ", lpad("$r", maxl, " "), " → ")
            print_exog(v)
        end
        print(io, "\n   ⋮")
        for (r, v) in cp[bot:end]
            print(io, "\n  ", lpad("$r", maxl, " "), " → ")
            print_exog(v)
        end
    end
end

#######################################
# The user interface to modify plan instances.

"""
    setplanvalue!(plan, value, vars, range)
    setplanvalue!(plan, value, vars, date)
    setplanvalue!(plan, value, vars, dates)

Modify the status of the given variable(s) on the given date(s). If `value` is
`true` then variables become exogenous, otherwise they become endogenous.

"""
function setplanvalue!(p::Plan{T}, val::Bool, vars::Array{Symbol,1}, date::AbstractUnitRange{T}) where T <: MIT
    firstindex(p) ≤ first(date) && last(date) ≤ lastindex(p) || throw(BoundsError(p, date))
    idx1 = _offset(p, date)
    for v in vars
        idx2 = get(p.varshks, v, 0)
        if idx2 == 0
            throw(ArgumentError("Unknown variable or shock name: $v"))
        end
        p.exogenous[idx1, idx2] .= val
    end
    return p
end
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::AbstractArray{Symbol,1}, date::AbstractUnitRange{Int}) = setplanvalue!(p, val, vars, UnitRange{MIT{Unit}}(date))
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::AbstractArray{Symbol,1}, date::Integer) = setplanvalue!(p, val, vars, date*U:date*U)
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::AbstractArray{Symbol,1}, date::MIT{Unit}) = setplanvalue!(p, val, vars, date:date)
setplanvalue!(p::Plan{T}, val::Bool, vars::AbstractArray{Symbol,1}, date::T) where T <: MIT = setplanvalue!(p, val, vars, date:date)
setplanvalue!(p::Plan, val::Bool, vars::AbstractArray{Symbol,1}, date) = (foreach(d -> setplanvalue!(p, val, vars, d), date); p)


"""
    exogenize!(plan, vars, date)

Modify the given plan so that the given variables will be exogenous on the given
dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such. `date` can
be a moment in time (same type as the plan), or a range or an iterable or a
container.

"""
exogenize!(p::Plan, var, date) = setplanvalue!(p, true, Symbol[var,], date)
exogenize!(p::Plan, vars::AbstractVector, date) = setplanvalue!(p, true, Symbol[vars...], date)
exogenize!(p::Plan, vars::Tuple, date) = setplanvalue!(p, true, Symbol[vars...], date)


"""
    endogenize!(plan, vars, date)

Modify the given plan so that the given variables will be endogenous on the
given dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such.
`date` can be a moment in time (same type as the plan), or a range or an
iterable or a container.

"""
endogenize!(p::Plan, var, date) = setplanvalue!(p, false, Symbol[var,], date)
endogenize!(p::Plan, vars::AbstractVector, date) = setplanvalue!(p, false, Symbol[vars...], date)
endogenize!(p::Plan, vars::Tuple, date) = setplanvalue!(p, false, Symbol[vars...], date)


"""
    exog_endo!(plan, exog_vars, endo_vars, date)

Modify the given plan so that the given variables listed in `exog_vars` will be
exogenous and the variables listed in `endo_vars` will be endogenous on the
given dates. `exog_vars` and `endo_vars` can each be a `Symbol` or a `String` or
a `Vector` of such. `date` can be a moment in time (same type as the plan), or a
range or an iterable or a container.

"""
function exog_endo!(p::Plan, exog, endo, date)
    setplanvalue!(p, true, Symbol[exog...], date)
    setplanvalue!(p, false, Symbol[endo...], date)
end
function exog_endo!(p::Plan, exog::Union{AbstractString,Symbol,ModelVariable}, endo::Union{AbstractString,Symbol,ModelVariable}, date)
    setplanvalue!(p, true,Symbol[exog], date)
    setplanvalue!(p, false,Symbol[endo], date)
end

"""
    endo_exog!(plan, endo_vars, exog_vars, date)

Modify the given plan so that the given variables listed in `exog_vars` will be
exogenous and the variables listed in `endo_vars` will be endogenous on the
given dates. `exog_vars` and `endo_vars` can each be a `Symbol` or a `String` or
a `Vector` of such. `date` can be a moment in time (same type as the plan), or a
range or an iterable or a container.

"""
@inline endo_exog!(p::Plan, endo, exog, date) = exog_endo!(p, exog, endo, date)

"""
autoexogenize!(plan, model, date)

Modify the given plan according to the "autoexogenize" protocol defined in the
given model. All variables in the autoexogenization list become endogenous and
their corresponding shocks become exogenous over the given date or range. `date`
can be a moment in time (same frequency as the given plan), a range, an
iterable, or a container.

"""
function autoexogenize!(p::Plan, m::Model, date)
    auto_vars = keys(m.autoexogenize)
    auto_shks = values(m.autoexogenize)
    exog_endo!(p, auto_vars, auto_shks, date)
end

#######################################
# The user interface to prepare data for simulation.

@inline TimeSeriesEcon.frequencyof(p::Plan) = frequencyof(p.range)
@inline TimeSeriesEcon.firstdate(p::Plan) = first(p.range)
@inline TimeSeriesEcon.lastdate(p::Plan) = last(p.range)

#######################################
# The internal interface to simulations code.

"""
    plansum(model, plan)

Return the total number of exogenous variables in the simulation plan. Periods
over which initial and final conditions are imposed are not counted in this sum.

"""
plansum(m::Model, p::Plan) = sum(p.exogenous[(1 + m.maxlag):(end - m.maxlead), :])

export setexog!
"""
    setexog!(plan, t, vinds)

Modify the plan at time t such that `vinds` are exogenous and the rest are
endogenous.

"""
function setexog!(p::Plan, tt::Int, vinds)
    p.exogenous[tt, :] .= false
    p.exogenous[tt, vinds] .= true
end
setexog!(p::Plan{T}, tt::T, vinds) where T <: MIT = setexog!(p, _offset(p, tt), vinds)

end # module Plans

using .Plans

export Plan,
    exogenize!, endogenize!, exog_endo!, endo_exog!, autoexogenize!
