

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
    zeroarray, zerodict, steadystatearray, steadystatedict

# exported from this module, but meant for internal use in StateSpaceEcon
export plansum, makeexogenizedata


"""
    Plan{T}

A data structure representing the simulation plan. It holds information about
the time range of the simulation and which variables/shocks are exogenous at
each period.

"""
struct Plan{T} <: AbstractVector{Vector{Symbol}}
    range::AbstractRange{T}
    varsshks::NamedTuple
    exogenous::BitArray{2}
end

"""
    Plan(model, range)

Create a default simulation plan for the given model over the given range. The
range of the plan is augmented to include periods before and after the given
range, over which initial and final conditions will be applied.

"""
function Plan(model::Model, range::AbstractRange)
    range = (first(range) - model.maxlag):(last(range) + model.maxlead)
    nvars = ModelBaseEcon.nvariables(model)
    nshks = ModelBaseEcon.nshocks(model)
    vs_names = tuple(ModelBaseEcon.variables(model)..., ModelBaseEcon.shocks(model)...)
    varsshks = NamedTuple{vs_names}(1:(nvars + nshks))
    return Plan{eltype(range)}(range, varsshks, BitArray(var ≤ nvars for _ in range, var = 1:(nvars + nshks)))
end

#######################################
# AbstractVector interface 

Base.size(p::Plan) = size(p.range)
Base.axes(p::Plan) = (p.range,)
Base.length(p::Plan) = length(p.range)
Base.IndexStyle(::Plan) = IndexLinear()

function Base.getindex(p::Plan{T}, idx::T) where T 
    # error("")
    if first(p.range) ≤ idx ≤ last(p.range)
        vals = view(p.exogenous, idx - first(p.range) + 1, :)
        return Symbol[k for (k, v) in pairs(p.varsshks) if vals[v]]
    else
        throw(BoundsError(p, idx))
    end
end

function Base.getindex(p::Plan{T}, rng::AbstractRange{T}) where T
    rng.start < p.range.start && throw(BoundsError(p, rng.start))
    rng.stop > p.range.stop && throw(BoundsError(p, rng.stop))
    return Plan{T}(rng, p.varsshks, p.exogenous[1 - p.range.start .+ rng, :])
end

function Base.getindex(p::Plan{T}, rng::AbstractRange{T}, m::Model) where T
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
function setplanvalue!(p::Plan{T}, val::Bool, vars::Array{Symbol,1}, date::AbstractRange{T}) where T
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
setplanvalue!(p::Plan{T}, val::Bool, vars::Array{Symbol,1}, date::T) where T = setplanvalue!(p, val, vars, date:date)
setplanvalue!(p::Plan, val::Bool, vars::Array{Symbol,1}, dates) = (foreach(d->setplanvalue!(p, val, vars, d), dates); p)


"""
    exogenize!(plan, vars, date)

Modify the given plan so that the given variables will be exogenous on the given
dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such. `date` can
be a moment in time (same type as the plan), or a range or an iterable or a
container.

"""
exogenize!(p::Plan{T}, var::Symbol, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, true, [var,], date)
exogenize!(p::Plan{T}, var::AbstractString, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, true, [Symbol(var),], date)
exogenize!(p::Plan{T}, vars::Vector{<:AbstractString}, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, true, map(Symbol, vars), date)
exogenize!(p::Plan{T}, vars::Vector{Symbol}, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, true, vars, date)


"""
    endogenize!(plan, vars, date)

Modify the given plan so that the given variables will be endogenous on the
given dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such.
`date` can be a moment in time (same type as the plan), or a range or an
iterable or a container.

"""
endogenize!(p::Plan{T}, var::Symbol, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, false, [var,], date)
endogenize!(p::Plan{T}, var::AbstractString, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, false, [Symbol(var),], date)
endogenize!(p::Plan{T}, vars::Vector{<:AbstractString}, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, false, map(Symbol, vars), date)
endogenize!(p::Plan{T}, vars::Vector{Symbol}, date::Union{T,AbstractRange{T}}) where T = setplanvalue!(p, false, vars, date)


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
    endo_exog!(p, auto_vars, auto_shks, date)
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

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref), [`steadystatedict`](@ref)

"""
zeroarray(::Model, p::Plan) = zeros(Float64, size(p.exogenous))
zeroarray(m::Model, rng::AbstractRange) = zeroarray(m, Plan(m, rng))

"""
    zerodict(model, plan)
    zerodict(model, range)

Create a dictionary containing a Series of the appropriate range for each
variable in the model for a simulation with the given plan or over the given
range. If a range is given rather than a plan, the data is prepared for the
default plan over that range. This means that appropriate number of periods are
added before and after the range to account for initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref), [`steadystatedict`](@ref)

"""
zerodict(::Model, p::Plan) = Dict(string(v) => Series(p.range, 0.0) for v in keys(p.varsshks))
zerodict(m::Model, rng::AbstractRange) = zerodict(m, Plan(m, rng))

"""
    steadystatearray(model, plan)
    steadystatearray(model, range)

Create a matrix of the proper dimensions for a simulation with the given model
with the given plan or over the given range. The matrix is initialized with the
steady state level of each variable. If a range is given rather than a plan, it
is augmented with periods before and after the given range in order to
accommodate initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref), [`steadystatedict`](@ref)

"""
steadystatearray(m::Model, rng::AbstractRange) = steadystatearray(m, Plan(m, rng))
steadystatearray(m::Model, p::Plan) = Float64[i <= ModelBaseEcon.nvariables(m) ? m.sstate[v] : 0.0 for _ in p.range, (v, i) = pairs(p.varsshks)]

"""
    steadystatearray(model, plan)
    steadystatearray(model, range)

Create a dictionary containing a Series of the appropriate range for each
variable in the model for a simulation with the given plan or over the given
range. The matrix is initialized with the steady state level of each variable.
If a range is given rather than a plan, it is augmented with periods before and
after the given range in order to accommodate initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref), [`steadystatedict`](@ref)

"""
steadystatedict(m::Model, rng::AbstractRange) = steadystatedict(m, Plan(m, rng))
steadystatedict(m::Model, p::Plan) = Dict(string(v) => Series(p.range, i <= ModelBaseEcon.nvariables(m) ? m.sstate[v] : 0.0) for (v, i) in pairs(p.varsshks))

#######################################
# The internal interface to simulations code.

"""
    plansum(model, plan)

Return the total number of exogenous variables in the simulation plan. Periods
over which initial and final conditions are imposed are not counted in this sum.

"""
plansum(m::Model, p::Plan) = sum(p.exogenous[(1 + m.maxlag):(end - m.maxlead), :])

"""
    makeexogenizedata(model, plan, t::Int, data)

Extract from the `data` vector the values of the exogenous variables at the
given time. `data` must be a 1-dimensional vector, i.e. only the row
corresponding to the given date. `t` must be integer, regardless of the range
type of the simulation. Value of 1 corresponds to the first period of the plan
(the plan contains initial conditions, so the first period of the simulation has
Integer index m.maxlag+1).

"""
function makeexogenizedata(m::Model, p::Plan, t::Int64, data)
    names = p[firstindex(p) + t - 1]
    values = data[p.exogenous[t,:]]
    return NamedTuple{tuple(names...)}(values)
end


end # module Plans

using .Plans
export Plan,
    exogenize!, endogenize!, exog_endo!, endo_exog!, autoexogenize!,
    zeroarray, zerodict, steadystatearray, steadystatedict

