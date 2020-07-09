
"""
    update_auxvars_ss()
"""
function update_auxvars_ss(point::Vector{Float64}, model::Model)
    nvars = length(model.variables)
    nshks = length(model.shocks)
    nauxs = length(model.auxvars)
    
    if nauxs == 0
        return point
    end
    
    trange = -model.maxlag:model.maxlead
    ntimes = length(trange)
    time0 = 1 + model.maxlag

    pt0 = [ones(ntimes) 0.0 .+ trange] * reshape(point, 2, :)
    pt0 = [pt0[:,1:nvars] zeros(ntimes, nshks)]
    pt0 = ModelBaseEcon.update_auxvars(pt0, model)

    result = Vector{Float64}(undef, 2 * (nvars + nauxs))
    result[1:length(point)] = point
    for i = 1:nauxs
        result[2 * (nvars + i) - 1] = pt0[time0, nvars + nshks + i]
    end

    if ! model.flags.ssZeroSlope
        shift = model.options.shift
        pt1 = [ones(ntimes) shift .+ trange] * reshape(point, 2, :)
        pt1 = [pt1[:,1:nvars] zeros(ntimes, nshks)]
        pt1 = ModelBaseEcon.update_auxvars(model, pt1)
        for i = 1:nauxs
            result[2 * (nvars + i)] = (pt1[time0, nvars + nshks + i] - pt0[time0, nvars + nshks + i]) / shift
        end
    end

    return result
end
@assert precompile(update_auxvars_ss, (Vector{Float64}, Model))



"""
    _do_update_auxvars_presolve!(model)

Call `update_auxvars_ss`, then call `presolve_sstate!`.

!!! warning
    This function is for internal use. Do not call directly.
"""
function _do_update_auxvars_presolve!(model::Model; verbose::Bool)
    ss = model.sstate
    if model.flags.ssZeroSlope
        ss.values[2:2:end] .= 0.0
        ss.mask[2:2:end] .= true
    end
    # sometimes update_auxvars_ss might change the the behaviour of presolve_sstate!
    # because it might set the values of aux variable differently and so
    # the presolve would be done on a different line parallel to the presolve-variable coordinate.
    aux_vals = update_auxvars_ss(ss.values, model)
    ss.values[ .! ss.mask ] = aux_vals[ .! ss.mask ]
    old_solved = -1
    solved = sum(ss.mask)
    while old_solved < solved
        presolve_sstate!(model; verbose = verbose)
        aux_vals .= update_auxvars_ss(ss.values, model)
        ss.values[ .! ss.mask ] = aux_vals[ .! ss.mask ]
        old_solved = solved
        solved = sum(ss.mask)
    end
    return nothing
end
@assert precompile(_do_update_auxvars_presolve!, (Model,))

"""
    clear_sstate!(model; lvl=0.1, slp=0.0, <options>)

Set the steady state values to the provided defaults and presolve.

### Arguments
  * `model` - the model instance
  * `lvl` - the default level value. Each steady state level is set to this number.
  * `slp` - the default slope value. Each steady state slope is set to this number.
"""
function clear_sstate!(model::Model; lvl = 0.1, slp = 0.0, verbose = model.options.verbose)
    ss = model.sstate
    foo = 2 * length(model.variables)
    ss.values[1:2:foo] .= lvl  # default initial guess for level
    ss.values[foo + 1:2:end] .= 0.0
    ss.values[2:2:end] .= slp  # default initial guess for slope
    ss.mask[:] .= false
    return _do_update_aux_presolve!(model; verbose = verbose)
end
export clear_sstate!
@assert precompile(clear_sstate!, (Model,))

"""
    initial_sstate!(model, init; <options>)

Set the steady state values from the given vector and presolve.

Call this function to specify initial guesses for the iterative steady state solver.
If the value of a steady state variable is known, add that as a steady state equation.

### Arguments
  * `model` - the model.
  * `init` - a vector of length equal to twice the number of variables in the model.
  The level and slope values are staggered, i.e., the level and slope of variable j 
  are in init[2j-1] and init[2j].
"""
function initial_sstate!(model::Model, init::AbstractVector{Float64}; verbose = model.options.verbose)
    ss = model.sstate
    nvars = length(model.variables)
    nauxs = length(model.auxvars)
    ninit = length(init)
    if ninit ∉ (2 * nvars + 2 * nauxs, 2 * nvars)
        error("Incorrect dimension if initial guess: $(ninit). Expected $(2 * nvars) or $(2 * nvars + 2 * nauxs)")
    end
    ss.values[1:ninit] = init
    ss.values[ninit + 1:end] .= 0.0
    ss.mask[:] .= false
    return _do_update_aux_presolve!(model; verbose = verbose)
end
export initial_sstate!
@assert precompile(initial_sstate!, (Model, Vector{Float64}))


