

"""
    Plans

Module part of StateSpaceEcon. This module implements the `Plan` data structure,
which is used in simulations. The plan object contains information about the
range of the simulation and which variables and shocks are exogenous or
endogenous at each period of the range.

### Constructor
  * [`Plan`](ref)`(model, range)`

### Modify the plan
  * [`exogenize!`](@ref), [`endogenize!`](@ref) - make variables exogenous or
    endogenous
  * [`exog_endo!`](@ref), [`endo_exog!`](@ref) - swap exogenous and endogenous
    variables
  * [`autoexogenize!`](@ref) - exogenize and endogenize variables according to
    the list in the model

### Prepare data for simulation
  * [`zeroarray`](@ref), [`zerodict`] - prepare a matrix or a dictionary of data
    for the simulation
  * [`steadystatearray`](@ref), [`steadystatedict`] - prepare a matrix or a
    dictionary of data for the simulation containing the steady state

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
    Plan{T}

A data structure representing the simulation plan. It holds information about
the time range of the simulation and which variables/shocks are exogenous at
each period.

"""
struct Plan{T <: MIT} <: AbstractVector{Vector{Symbol}}
    range::AbstractUnitRange{T}
    varsshks::NamedTuple
    exogenous::BitArray{2}
end

"""
    Plan(model, range)

Create a default simulation plan for the given model over the given range. The
range of the plan is augmented to include periods before and after the given
range, over which initial and final conditions will be applied.

"""
function Plan(model::Model, range::AbstractUnitRange)
    if !(eltype(range) <: MIT)
        range = UnitRange{MIT{Unit}}(range)
    end
    range = (first(range) - model.maxlag):(last(range) + model.maxlead)
    nvars = ModelBaseEcon.nvariables(model)
    nshks = ModelBaseEcon.nshocks(model)
    vs_names = tuple(ModelBaseEcon.variables(model)..., ModelBaseEcon.shocks(model)...)
    varsshks = NamedTuple{vs_names}(1:(nvars + nshks))
    return Plan{eltype(range)}(range, varsshks, BitArray(var > nvars for _ in range, var = 1:(nvars + nshks)))
end

#######################################
# AbstractVector interface 

Base.size(p::Plan) = size(p.range)
Base.axes(p::Plan) = (p.range,)
Base.length(p::Plan) = length(p.range)
Base.IndexStyle(::Plan) = IndexLinear()

@inline _offset(p::Plan{T}, idx::T) where T = 1 - first(p.range) + idx
@inline _offset(p::Plan{T}, idx::AbstractUnitRange{T}) where T = 1 - first(p.range) .+ idx

## Integer index is taken as is
Base.getindex(p::Plan, idx::Int) = p[p.range[idx]]
Base.getindex(p::Plan, idx::AbstractUnitRange{Int}) = p[p.range[idx]]
Base.getindex(p::Plan, idx::Int, ::Val{:inds}) = findall(p.exogenous[idx,:])

# Index of the frequency type returns the list of exogenous symbols
Base.getindex(p::Plan{T}, idx::T) where T <: MIT = [keys(p.varsshks)[p[_offset(p,idx),Val(:inds)]]...,]
function Base.getindex(p::Plan{T}, idx::T, ::Val{:inds}) where T <: MIT
    first(p.range) ≤ idx ≤ last(p.range) || throw(BoundsError(p, idx))
    return p[_offset(p, idx), Val(true)]
end

# A range returns the plan trimmed over that exact range.
Base.getindex(p::Plan{MIT{Unit}}, rng::AbstractUnitRange{Int}) = p[UnitRange{MIT{Unit}}(rng)]
function Base.getindex(p::Plan{T}, rng::AbstractUnitRange{T}) where T <: MIT
    rng.start < p.range.start && throw(BoundsError(p, rng.start))
    rng.stop > p.range.stop && throw(BoundsError(p, rng.stop))
    return Plan{T}(rng, p.varsshks, p.exogenous[_offset(rng), :])
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

function Base.show(io::IO, ::MIME"text/plain", p::Plan)
    # 0) show summary before setting :compact
    summary(io, p)
    isempty(p) && return
    print(io, ":")
    nrow, ncol = displaysize(io)
    if length(p) <= nrow - 5
        for r in p.range
            print(io, "\n  ", r, " => ", p[r])
        end
    else
        top = div(nrow - 5, 2)
        bot = length(p.range) - nrow + 6 + top
        for r in p.range[1:top]
            print(io, "\n  ", r, " => ", p[r])
        end
        print(io, "\n   ⋮")
        for r in p.range[bot:end]
            print(io, "\n  ", r, " => ", p[r])
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
    idx1 = 1 - firstindex(p) .+ date
    for v in vars
        idx2 = get(p.varsshks, v, 0)
        if idx2 == 0
            throw(ArgumentError("Unknown variable or shock name: $v"))
        end
        p.exogenous[idx1, idx2] .= val
    end
    return p
end
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::Array{Symbol,1}, date::AbstractUnitRange{Int}) = setplanvalue!(p, val, vars, UnitRange{MIT{Unit}}(date))
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::Array{Symbol,1}, date::Integer) = setplanvalue!(p, val, vars, ii(date):ii(date))
setplanvalue!(p::Plan{T}, val::Bool, vars::Array{Symbol,1}, date::T) where T <: MIT = setplanvalue!(p, val, vars, date:date)
setplanvalue!(p::Plan, val::Bool, vars::Array{Symbol,1}, date) = (foreach(d->setplanvalue!(p, val, vars, d), date); p)


"""
    exogenize!(plan, vars, date)

Modify the given plan so that the given variables will be exogenous on the given
dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such. `date` can
be a moment in time (same type as the plan), or a range or an iterable or a
container.

"""
exogenize!(p::Plan, var::Symbol, date) = setplanvalue!(p, true, [var,], date)
exogenize!(p::Plan, var::AbstractString, date) = setplanvalue!(p, true, [Symbol(var),], date)
exogenize!(p::Plan, vars::Vector{<:AbstractString}, date) = setplanvalue!(p, true, map(Symbol, vars), date)
exogenize!(p::Plan, vars::Vector{Symbol}, date) = setplanvalue!(p, true, vars, date)


"""
    endogenize!(plan, vars, date)

Modify the given plan so that the given variables will be endogenous on the
given dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such.
`date` can be a moment in time (same type as the plan), or a range or an
iterable or a container.

"""
endogenize!(p::Plan, var::Symbol, date) = setplanvalue!(p, false, [var,], date)
endogenize!(p::Plan, var::AbstractString, date) = setplanvalue!(p, false, [Symbol(var),], date)
endogenize!(p::Plan, vars::Vector{<:AbstractString}, date) = setplanvalue!(p, false, map(Symbol, vars), date)
endogenize!(p::Plan, vars::Vector{Symbol}, date) = setplanvalue!(p, false, vars, date)


"""
    exog_endo!(plan, exog_vars, endo_vars, date)

Modify the given plan so that the given variables listed in `exog_vars` will be
exogenous and the variables listed in `endo_vars` will be endogenous on the
given dates. `exog_vars` and `endo_vars` can each be a `Symbol` or a `String` or
a `Vector` of such. `date` can be a moment in time (same type as the plan), or a
range or an iterable or a container.

"""
function exog_endo!(p::Plan, exog, endo, date)
    setplanvalue!(p, true, exog, date)
    setplanvalue!(p, false, endo, date)
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
can be a moment in time (same type as the plan), or a range or an iterable or a
container.

"""
function autoexogenize!(p::Plan, m::Model, date)
    auto_vars = collect(keys(m.autoexogenize))
    auto_shks = collect(values(m.autoexogenize))
    exog_endo!(p, auto_vars, auto_shks, date)
end

#######################################
# The user interface to prepare data for simulation.

"""
    zeroarray(model, plan)
    zeroarray(model, range)

Create a matrix of the proper dimension for a simulation with the given model
with the given plan or over the given range. If a range is given, the data is prepared for the
default plan. This means that appropriate number of periods are added before and
after the range to account for initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref),
[`steadystatedict`](@ref)

"""
zeroarray(::Model, p::Plan) = zeros(Float64, size(p.exogenous))
zeroarray(m::Model, rng::AbstractUnitRange) = zeroarray(m, Plan(m, rng))

"""
    zerodict(model, plan)
    zerodict(model, range)

Create a dictionary containing a [`TSeries`](@ref) of the appropriate range for
each variable in the model for a simulation with the given plan or over the
given range. If a range is given rather than a plan, the data is prepared for
the default plan over that range. This means that appropriate number of periods
are added before and after the range to account for initial and final
conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref),
[`steadystatedict`](@ref)

"""
zerodict(::Model, p::Plan) = Dict(string(v) => TSeries(p.range, 0.0) for v in keys(p.varsshks))
zerodict(m::Model, rng::AbstractUnitRange) = zerodict(m, Plan(m, rng))

"""
    zerodata(model, plan)
    zerodata(model, range)

Create a `NamedTuple` containing a [`TSeries`](@ref) of the appropriate range for each
variable in the model for a simulation with the given plan or over the given
range. If a range is given rather than a plan, the data is prepared for the
default plan over that range. This means that appropriate number of periods are
added before and after the range to account for initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref),
[`steadystatedict`](@ref)

"""
zerodata(::Model, p::Plan) = NamedTuple{keys(p.varsshks)}(((TSeries(p.range, 0.0) for _ in p.varsshks)...,))
zerodata(m::Model, rng::AbstractUnitRange) = zerodata(m, Plan(m, rng))

"""
    steadystatearray(model, plan)
    steadystatearray(model, range)

Create a matrix of the proper dimensions for a simulation with the given model
with the given plan or over the given range. The matrix is initialized with the
steady state level of each variable. If a range is given rather than a plan, it
is augmented with periods before and after the given range in order to
accommodate initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref),
[`steadystatedict`](@ref)

"""
steadystatearray(m::Model, rng::AbstractUnitRange) = steadystatearray(m, Plan(m, rng))
steadystatearray(m::Model, p::Plan) = Float64[i <= ModelBaseEcon.nvariables(m) ? m.sstate[v] : 0.0 for _ in p.range, (v, i) = pairs(p.varsshks)]

"""
    steadystatearray(model, plan)
    steadystatearray(model, range)

Create a dictionary containing a [`TSeries`](@ref) of the appropriate range for each
variable in the model for a simulation with the given plan or over the given
range. The matrix is initialized with the steady state level of each variable.
If a range is given rather than a plan, it is augmented with periods before and
after the given range in order to accommodate initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref),
[`steadystatedict`](@ref)

"""
steadystatedict(m::Model, rng::AbstractUnitRange) = steadystatedict(m, Plan(m, rng))
steadystatedict(m::Model, p::Plan) = Dict(string(v) => TSeries(p.range, i <= ModelBaseEcon.nvariables(m) ? m.sstate[v] : 0.0) for (v, i) in pairs(p.varsshks))

"""
    steadystatedata(model, plan)
    steadystatedata(model, range)

Create a dictionary containing a [`TSeries`](@ref) of the appropriate range for each
variable in the model for a simulation with the given plan or over the given
range. The matrix is initialized with the steady state level of each variable.
If a range is given rather than a plan, it is augmented with periods before and
after the given range in order to accommodate initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref),
[`steadystatedict`](@ref)

"""
steadystatedata(m::Model, rng::AbstractUnitRange) = steadystatedict(m, Plan(m, rng))
steadystatedata(m::Model, p::Plan) = NamedTuple{keys(p.varsshks)}(((TSeries(p.range, i ≤ ModelBaseEcon.nvariables(m) ? m.sstate[v] : 0) for (v, i) in pairs(p.varsshks))...,))

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
    exogenize!, endogenize!, exog_endo!, endo_exog!, autoexogenize!,
    zeroarray, zerodict, zerodata, steadystatearray, steadystatedict, steadystatedata
