##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

# the following functions use the Plan object to create simulation data

export zeroarray, zerodict, zerodata, steadystatearray, steadystatedict, steadystatedata

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
function zeroarray end
@inline zeroarray(m::Model, rng::AbstractUnitRange) = zeroarray(m, Plan(m, rng))
@inline zeroarray(m::Model, p::Plan) = inverse_transform(zeros(Float64, size(p.exogenous)), m)

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
function zerodict end
@inline zerodict(m::Model, rng::AbstractUnitRange) = zerodict(m, Plan(m, rng))
function zerodict(m::Model, p::Plan) 
    data = Dict{String,Any}()
    for v in m.varshks
        push!(data, string(v.name) => TSeries(p.range, inverse_transform(0.0, v)))
    end
    return data
end

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
function zerodata end
@inline zerodata(m::Model, rng::AbstractUnitRange) = zerodata(m, Plan(m, rng))
@inline zerodata(m::Model, p::Plan) = SimData(firstdate(p), m.varshks, zeroarray(m, p))

##################

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
function steadystatearray end
@inline steadystatearray(m::Model, rng::AbstractUnitRange; ref=first(rng)) = steadystatearray(m, Plan(m, rng), ref=ref)
function steadystatearray(m::Model, p::Plan; ref=firstdate(p) + m.maxlag)
    vs = keys(p.varshks) 
    data = Matrix{Float64}(undef, length(p), length(vs))
    for (vi, v) ∈ enumerate(vs)
        data[:, vi] = m.sstate.:($v)[p.range, ref=ref]
    end
    return data
end

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
function steadystatedict end
@inline steadystatedict(m::Model, rng::AbstractUnitRange; ref=first(rng)) = steadystatedict(m, Plan(m, rng), ref=ref)
function steadystatedict(m::Model, p::Plan; ref=firstdate(p) + m.maxlag)
    data = Dict{String,Any}()
    for v ∈ keys(p.varshks)
        push!(data, string(v) => TSeries(p.range, m.sstate.:($v)[p.range, ref=ref]))
    end
    return data
end

"""
    steadystatedata(model, plan)
    steadystatedata(model, range)

Create a [`SimData`](@ref) containing a [`TSeries`](@ref) of the appropriate range for each
variable in the model for a simulation with the given plan or over the given
range. The matrix is initialized with the steady state level of each variable.
If a range is given rather than a plan, it is augmented with periods before and
after the given range in order to accommodate initial and final conditions.

See also: [`zeroarray`](@ref), [`zerodict`](@ref), [`steadystatearray`](@ref),
[`steadystatedict`](@ref)

"""
function steadystatedata end
@inline steadystatedata(m::Model, rng::AbstractUnitRange; ref=first(rng)) = steadystatedata(m, Plan(m, rng), ref=ref)
function steadystatedata(m::Model, p::Plan; ref=firstdate(p) + m.maxlag) 
    return SimData(firstdate(p), m.varshks, steadystatearray(m, p, ref=ref))
end
