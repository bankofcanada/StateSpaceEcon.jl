##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

# the following functions use the Plan object to create simulation data

export zeroarray, zerodict, zerodata, zeroworkspace
export steadystatearray, steadystatedict, steadystatedata, steadystateworkspace

"""
    zeroarray(model, plan)
    steadystatearray(model, plan; [ref=firstdate(plan) + m.maxlag)

Create a `Matrix{Float64}` of the proper dimension for a simulation with the
given model with the given plan. It is initialized to 0 or the steady state.

This function returns a `Matrix`. We recommend using [`zerodata`](@ref). See
also [`zeroworkspace`](@ref)

* In the case of steady state solution that is not stationary in time (i.e.,
  constant rate of change or constant rate of growth) use the `ref` option to
  specify the period in which the steady state level is given. The default is
  the first simulation period.

!!! note "Deprecation Note"

    `zeroarray(model, range)` will be removed in future versions. Always create
    a simulation `Plan` explicitly.

"""
function zeroarray end, function steadystatearray end

@deprecate zeroarray(m::Model, rng::AbstractUnitRange) zeroarray(m, Plan(m, rng))
@deprecate steadystatearray(m::Model, rng::AbstractUnitRange; ref = first(rng)) steadystatearray(m, Plan(m, rng), ref = ref)

@inline zeroarray(m::Model, p::Plan) = inverse_transform(zeros(Float64, size(p.exogenous)), m)
function steadystatearray(m::Model, p::Plan; ref = firstdate(p) + m.maxlag)
    return hcat((m.sstate[v][p.range, ref = ref] for v in m.varshks)...)
end

##################

"""
    zerodict(model, plan)
    steadystatedict(model, plan; [ref=firstdate(plan) + m.maxlag))

!!! note "Deprecation Note"

    This function will be removed in a future version. Use
    [`zeroworkspace`](@ref)`(model, plan)`.

"""
function zerodict end
@deprecate zerodict(m::Model, rng::AbstractUnitRange) zeroworkspace(m, Plan(m, rng))
@deprecate steadystatedict(m::Model, rng::AbstractUnitRange) steadystateworkspace(m, Plan(m, rng))
@deprecate zerodict(m::Model, p::Plan) zeroworkspace(m, p)
@deprecate steadystatedict(m::Model, p::Plan) steadystateworkspace(m, p)

##################

"""
    zeroworkspace(model, plan)
    steadystateworkspace(model, plan; [ref=firstdate(plan) + m.maxlag))

Create a [`TimeSeriesEcon.Workspace`](@ref) containing a `TSeries` for each
variable/shock in the given `model`. They are initialized to 0 or the steady state solution.

* In the case of steady state solution that is not stationary in time (i.e.,
  constant rate of change or constant rate of growth) use the `ref` option to
  specify the period in which the steady state level is given. The default is
  the first simulation period.

We recommend using [`zerodata`](@ref). See also [`zeroarray`](@ref).
"""
function zeroworkspace(model::Model, plan::Plan)
    rng = plan.range
    return Workspace(v.name => TSeries(rng, inverse_transform(0.0, v)) for v = model.varshks)
end
function steadystateworkspace(model::Model, plan::Plan; ref = firstdate(plan) + model.maxlag)
    rng = plan.range
    return Workspace(v.name => TSeries(rng, model.sstate[v][rng, ref = ref]) for v = model.varshks)
end

##################

"""
    zerodata(model, plan)

Create a [`SimData`] for a simulation with the given `model` and `plan`. Columns
correspond to the model variables and shocks in the correct order. Data is initialized
with 0 or the steady state. 

* In the case of steady state solution that is not stationary in time (i.e.,
  constant rate of change or constant rate of growth) use the `ref` option to
  specify the period in which the steady state level is given. The default is
  the first simulation period.

See also [`zeroarray`](@ref) and [`zeroworkspace`](@ref)
"""
function zerodata end
@deprecate zerodata(m::Model, rng::AbstractUnitRange) zerodata(m, Plan(m, rng))
@inline zerodata(m::Model, p::Plan) = SimData(p.range, m.varshks, zeroarray(m, p))
@deprecate steadystatedata(m::Model, rng::AbstractUnitRange; ref = firstdate(p) + m.maxlag) steadystatedata(m, Plan(m, rng); ref = ref)
@inline steadystatedata(m::Model, p::Plan; ref = firstdate(p) + m.maxlag) = SimData(p.range, m.varshks, steadystatearray(m, p; ref = ref))

##################

