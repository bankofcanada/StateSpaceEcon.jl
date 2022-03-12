##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    update_auxvars_ss(point, model)

This function is for internal use. Do not call directly.
"""
function update_auxvars_ss(point::Vector{Float64}, model::Model)
    nauxs = length(model.auxvars)

    if nauxs == 0
        return point
    end

    nvars = length(model.variables)
    nshks = length(model.shocks)
    nall = nvars + nshks + nauxs

    trange = -model.maxlag:model.maxlead
    ntimes = length(trange)
    time0 = 1 + model.maxlag

    pt0 = [ones(ntimes) 0.0 .+ trange] * reshape(point, 2, :)
    # pt0 = [pt0[:,1:nvars] zeros(ntimes, nshks)]
    pt0 = ModelBaseEcon.update_auxvars(pt0, model)

    result = Vector{Float64}(undef, 2 * nall)
    result[1:length(point)] = point
    # assign levels of aux variables
    for i = 1:nauxs
        result[2*(nvars+nshks+i)-1] = pt0[time0, nvars+nshks+i]
    end

    # assign slopes of aux variables, if not known to be zero
    if !model.flags.ssZeroSlope
        shift = model.options.shift
        pt1 = [ones(ntimes) shift .+ trange] * reshape(point, 2, :)
        # pt1 = [pt1[:,1:nvars] zeros(ntimes, nshks)]
        pt1 = ModelBaseEcon.update_auxvars(pt1, model)
        for i = 1:nauxs
            result[2*(nvars+nshks+i)] = (pt1[time0, nvars+nshks+i] - pt0[time0, nvars+nshks+i]) / shift
        end
    end

    return result
end
@assert precompile(update_auxvars_ss, (Vector{Float64}, Model))

function _do_warn(args...)
    println(args...)
end

"""
    _do_update_auxvars_presolve!(model)

Call `update_auxvars_ss`, then call `presolve_sstate!`.

This function is for internal use. Do not call directly.
"""
function _do_update_auxvars_presolve!(model::Model, verbose::Bool, method::Symbol = :bisect)
    ss = model.sstate
    # set shocks (level and slope) and steady slopes to 0.0
    for (i, v) in enumerate(model.allvars)
        if isshock(v)
            ss.values[2i.+(-1:0)] .= 0.0
            ss.mask[2i.+(-1:0)] .= true
        elseif issteady(v)
            ss.values[2i] = 0.0
            ss.mask[2i] = true
        end
    end
    # if all slopes are zero, make it so
    if model.flags.ssZeroSlope
        ss.values[2:2:end] .= 0.0
        ss.mask[2:2:end] .= true
    end
    # presolve only the steady state constraints (to apply the values of exog variables)
    if !isempty(ss.constraints)
        for eqn in ss.constraints
            ModelBaseEcon._update_eqn_params!(eqn.eval_resid, model.parameters)
        end
        presolve_sstate!(ss.constraints, ss.mask, ss.values; model.tol, verbose, method)
    end
    # make sure all exog variables are set to solved
    exogenous_not_given = Symbol[]
    for v in ss.vars
        if isexog(v.name)
            if !all(v.mask)
                push!(exogenous_not_given, v.name.name)
            end
            v.mask .= true
        end
    end
    if !isempty(exogenous_not_given)
        _do_warn("The following @exog variables do not have an assigned steady state. Use `@steadystate model exogvar = val`", exogenous_not_given)
    end
    # sometimes update_auxvars_ss might change the behaviour of presolve_sstate!
    # because it might set the values of aux variable differently and so
    # the presolve would be done on a different line parallel to the presolve-variable coordinate.
    aux_vals = update_auxvars_ss(ss.values, model)
    ss.values[.!ss.mask] = aux_vals[.!ss.mask]
    while presolve_sstate!(model; verbose, method)
        aux_vals .= update_auxvars_ss(ss.values, model)
        ss.values[.!ss.mask] = aux_vals[.!ss.mask]
    end
    return nothing
end
@assert precompile(_do_update_auxvars_presolve!, (Model, Bool, Symbol,))

"""
    clear_sstate!(model; lvl=0.1, slp=0.0, <options>)

Set the steady state values to the provided defaults and presolve.

### Arguments
  * `model` - the model instance
  * `lvl`, `slp` - the initial guess for the level and the slope. Each could be
    a number or a vector of length equal to the number of variable in the mode.

### Options
Standard options (default values are taken from `model.options`)
  * `verbose`
"""
function clear_sstate!(model::Model; lvl = 0.1, slp = 0.0, verbose = model.options.verbose)
    ss = model.sstate
    nvars = length(model.variables)
    nshks = length(model.shocks)
    fill!(ss.values, 0.0)
    ss.values[1:2:2nvars] .= lvl  # default initial guess for level
    ss.values[2:2:2nvars] .= slp  # default initial guess for slope
    ss.mask[:] .= false
    return _do_update_auxvars_presolve!(model, verbose)
end
export clear_sstate!
@assert precompile(clear_sstate!, (Model,))

"""
    initial_sstate!(model, init; <options>)

Set the steady state values from the given vector and presolve.

Call this function to specify initial guesses for the iterative steady state
solver. If the value of a steady state variable is known, it is better to use
[`@steadystate`](@ref ModelBaseEcon.@steadystate) to add that as a steady state
constraint.

### Arguments
  * `model` - the model.
  * `init` - a vector of length equal to twice the number of variables in the
    model. The level and slope values are staggered, i.e., the level and slope
    of variable j are in init[2j-1] and init[2j].

### Options
Standard options (default values are taken from `model.options`)
  * `verbose`

"""
function initial_sstate!(model::Model, init::AbstractVector{Float64}; verbose = model.options.verbose)
    ss = model.sstate
    nvars = length(model.variables)
    nshks = length(model.shocks)
    nauxs = length(model.auxvars)
    ninit = length(init)
    if ninit ∉ (2nvars, 2nvars + 2nauxs, 2nvars + 2nshks + 2nauxs)
        error("Incorrect dimension if initial guess: $(ninit). Expected $(2nvars) or $(2nvars + 2nauxs) or $(2nvars + 2nshks + 2nauxs)")
    end
    ss.values[1:ninit] = init
    ss.values[ninit+1:end] .= 0.0
    ss.mask[:] .= false
    return _do_update_auxvars_presolve!(model, verbose)
end
export initial_sstate!
@assert precompile(initial_sstate!, (Model, Vector{Float64}))


