
"""
    sim_nr!(x, sd, maxiter, tol, verbose [, sparse_solver])

Solve the simulation problem.
  * `x` - the array of data for the simulation. All initial, final and exogenous
    conditions are already in place.
  * `sd::AbstractSolverData` - the solver data constructed for the simulation
    problem.
  * `maxiter` - maximum number of iterations.
  * `tol` - desired accuracy.
  * `verbose` - whether or not to print progress information.
  * `sparse_solver` (optional) - a function called to solve the linear system A
    x = b for x. Defaults to A\\b

"""
function sim_nr!(x::AbstractArray{Float64}, sd::AbstractSolverData,
                maxiter::Int64, tol::Float64, verbose::Bool,
                sparse_solver::Function=(A, b) -> A \ b)
    for it = 1:maxiter
        @timer Fx, Jx = global_RJ(x, x, sd)
        @timer nFx = norm(Fx, Inf)
        if nFx < tol
            if verbose
                println("$it, || Fx || = $(nFx)")
            end
            break
        end
        @timer Δx = sparse_solver(Jx, Fx)
        @timer nΔx = norm(vec(Δx), Inf)
        assign_update_step!(x, -1.0, Δx, sd)
        if verbose
            println("$it, || Fx || = $(nFx), || Δx || = $(nΔx)")
        end
        if nΔx < tol
            break
        end
    end
    return nothing
end


export simulate
"""
    simulate(model, plan, data; <options>)

Run a simulation for the given model, simulation plan and exogenous data.

### Arguments
  * `model` - the [`Model`](@ref) instance to simulate.
  * `plan` - the [`Plan`](@ref) for the simulation.
  * `data` - a 2D `Array` containing the exogenous data. This includes the
    initial and final conditions.

### Options as keyword arguments
  * `fctype::`[`FCType`](@ref) - set the desired final condition type for the
    simulation. The default value is [`fcgiven`](@ref). Other possible values
    include [`fclevel`](@ref) and [`fcslope`](@ref).
  * `initial_guess::AbstractMatrix{Float64}` - a 2D `Array` containing the
    initial guess for the solution. This is used to start the Newton-Raphson
    algorithm. The default value is an empty array (`zeros(0,0)`), in which case
    we use the exogenous data for the initial condition. You can use the steady
    state solution using [`steadystatearray`](@ref).
  * `linearize::Bool` - set to `true` to instruct the solver to use the
    liearized model. If the model is already linearized, this option has the
    effect that the model gets linearized about the current steady stat and with
    the value of `deviation` given here. Otherwise the model is linearized about
    the steady state. After the simulation is computed, the model is restored to
    its original state. Default value is `false`.
  * `deviation::Bool` - set to `true` if the `data` is in deviations from the
    steady state. This is only relevant if the `linearize` option is set to
    `true`. Default value is `false`.
  * `anticipate::Bool` - set to `false` to instruct the solver that all shocks
    are unanticilated by the agents. Default value is `true`.
  * `verbose::Bool` - control whether or not to print progress information.
    Default value is taken from `model.options`.
  * `tol::Float64` - set the desired accuracy. Default value is taken from
    `model.options`.
  * `maxiter::Int` - algorithm fails if the desired accuracy is not reached
    within this maximum number of iterations. Default value is taken from
    `model.options`.

### See also: 

### Examples

"""
function simulate(m::Model, p::Plan, exog_data::AbstractArray{Float64,2};
                    initial_guess::AbstractArray{Float64,2}=zeros(0, 0),
                    linearize::Bool=false,
                    deviation::Bool=false,
                    anticipate::Bool=true,
                    verbose::Bool=m.options.verbose,
                    tol::Float64=m.options.tol,
                    maxiter::Int64=m.options.maxiter,
                    fctype::FCType=fcgiven,
                    expectation_horizon::Union{Nothing,Int64}=nothing,
                    sparse_solver::Function=(A, b) -> A \ b
    )
    NT = length(p.range)
    nauxs = length(m.auxvars)
    if size(exog_data) != (NT, length(m.varshks))
        error("Incorrect dimensions of exog_data. Expected $((NT, length(m.varshks))), got $(size(exog_data)).")
    end
    if !isempty(initial_guess) && size(initial_guess) != (NT, length(m.varshks))
        error("Incorrect dimensions of initial_guess. Expected $((NT, length(m.varshks))), got $(size(exog_data)).")
    end
    exog_data = hcat(exog_data, zeros(size(exog_data, 1), nauxs))
    if !isempty(initial_guess)
        x = @timer ModelBaseEcon.update_auxvars(initial_guess, m)
    else
        x = @timer ModelBaseEcon.update_auxvars(exog_data, m)
    end
    if linearize
        org_med = m.evaldata
        linearize!(m; deviation=deviation)
    end
    if anticipate
        @timer gdata = StackedTimeSolverData(m, p, fctype)
        @timer assign_exog_data!(x, exog_data, gdata)
        @timer sim_nr!(x, gdata, maxiter, tol, verbose, sparse_solver)
    else # unanticipated shocks
        init = 1:m.maxlag
        term = NT .+ (1 - m.maxlead:0)
        sim = 1 + m.maxlag:NT - m.maxlead
        nvars = length(m.variables)
        nshks = length(m.shocks)
        shkinds = nvars .+ (1:nshks)
        varshkinds = 1:(nvars + nshks)
        x[init, varshkinds] .= exog_data[init, varshkinds]
        # x[term, varshkinds] .= exog_data[term, varshkinds]
        x[sim, shkinds] .= 0.0
        t0 = first(sim)
        T = last(sim)
        if expectation_horizon === nothing
            # the original code, where we always simulate until the end with the true final conditions
            for t in sim
                exog_inds = p[t, Val(:inds)]
                psim = Plan(m, t:T)
                if (t != t0) && (psim[t0, Val(:inds)] == exog_inds) && (maximum(abs, exog_data[t,shkinds]) < tol)
                    continue
                end
                setexog!(psim, t0, exog_inds)
                @timer gdata = StackedTimeSolverData(m, psim, fctype)
                x[t,exog_inds] = exog_data[t,exog_inds]
                # @timer assign_exog_data!(x[psim.range,:], exog_data[psim.range,:], gdata)
                sim_range = UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                assign_final_condition!(xx, exog_data[sim_range,:], gdata, Val(gdata.FC))
                @timer sim_nr!(xx, gdata, maxiter, tol, verbose, sparse_solver)
            end
        else
            # the new code, where the first and last simulations use the true 
            # simulation range and final condition, while the intermediate 
            # simulations use expectation_horizon steps with fcslope
            if expectation_horizon == 0
                expectation_horizon = length(sim)
            elseif expectation_horizon < 10 * m.maxlead
                @warn "Expectation horizon may be too short for this model. Consider setting it to at least $(10 * m.maxlead)."
            end
            x = [x; zeros(expectation_horizon, size(x, 2))]
            ninit = length(init)
            nterm = length(term)
            # first simulation
            let t = t0  
                # first run is with the full range, the true fctype, 
                # and only the first period is imposed
                exog_inds = p[t, Val(:inds)]
                psim = Plan(m, t:T)
                setexog!(psim, t0, exog_inds)
                @timer sdata = StackedTimeSolverData(m, psim, fctype)
                sim_range = UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                assign_exog_data!(xx, exog_data[sim_range,:], sdata)
                sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver)
            end
            # intermediate simulations
            last_t::Int64 = t0
            psim = Plan(m, 0:expectation_horizon - 1)
            sdata = StackedTimeSolverData(m, psim, fcslope)
            for t in sim[2:end]
                exog_inds = p[t, Val(:inds)]
                # we need to run a simulation if a variable is exogenous, or if a shock value is not zero
                # these intermediate simulations are always with fcslope, 
                #       have length equal to expectation_horizon and 
                #       only the first period is imposed
                if (exog_inds == shkinds) && (maximum(abs, exog_data[t, shkinds]) <= tol)
                    continue
                end
                setexog!(psim, t0, exog_inds)
                update_plan!(sdata, m, psim)
                # note that the range always goes from 0 to expectation_horizon-1, 
                # so we need to add t in order to get the correct set of rows of x
                sim_range = t .+ UnitRange(psim.range)
                xx = view(x, sim_range, :)
                # The initial conditions are already set
                # The exogenous values are already set as well, except for the first period
                # In other words, we only need to impose the first period here
                @timer xx[t0, exog_inds] = exog_data[t, exog_inds]
                # Update the final conditions
                @timer assign_final_condition!(xx, zeros(0, 0), sdata, Val(fcslope))
                @timer sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver)
                last_t = t  # keep track of last simulation time
            end
            # last simulation
            if last_t > t0
                # do we need to re-run the last simulation?
                # if it didn't reach T, then yes
                # if the final condition is not fcslope, then yes
                if (last_t + expectation_horizon < T) || (fctype != fcslope)
                    psim = Plan(m, last_t + 1:T)
                    # there are no unanticipated shocks in this simulation
                    sdata = StackedTimeSolverData(m, psim, fctype)
                    # the initial conditions and the exogenous data are already in x
                    # we only need the final conditions
                    sim_range = UnitRange(psim.range)
                    xx = view(x, sim_range, :)
                    assign_final_condition!(xx, exog_data[sim_range, :], sdata, Val(fctype))
                    @timer sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver)
                end
            end
            x = x[1:end - expectation_horizon,:]
        end
    end
    if linearize
        m.evaldata = org_med
    end
    return x[:,1:end - nauxs]
end

# The versions of simulate with Dict

# Simulate command, IRIS style but without a range
function simulate(m::Model, D1::Dict{<:AbstractString,<:Any}, plan::Plan; 
    deviation::Bool=false, overlay::Bool=false, kwargs...)::Dict{String,<:Any}
    # Convert dictionary to Array{Float64,2}
    data01 = dict2array(D1, m.varshks, range=plan.range)
    # Adjust array with steady state values if necessary
    if deviation
        # Check that the steady state has been solved for
        if all(m.sstate.mask) && !any(isnan.(m.sstate.values)) && !any(isinf.(m.sstate.values))
            # Add steady state values to data01
            datass = sstatearray(m, plan);
            data01 = data01 .+ datass;
        else
            # The SS is not solved. We issue an error message.
            error("The steady state is not solved.")
        end
    end
    ig = zeros(0, 0)
    if :initial_guess ∈ keys(kwargs)
        if kwargs[:initial_guess] isa Dict
            ig = dict2array(kwargs[:initial_guess], m.varshks, range=plan.range)
        end
    end
    # Call native simulate commmand with Array{Float64,2}
    data02 = simulate(m, plan, data01; kwargs..., initial_guess=ig)
    # Remove steady state values from data02
    if deviation
        data02 = data02 .- datass;
    end
    # Reconvert Array{Float64,2} to Dict{String,Any}
    D2 = array2dict(data02, m.varshks, plan.range[1])
    # Overlay D1 and D2
    if overlay
        D3 = dictoverlay(D1, D2)
    else
        D3 = D2
    end
    return D3
end

# Simulate command, IRIS style with a range
function simulate(m::Model, D1::Dict{<:AbstractString,<:Any}, rng::AbstractUnitRange, plan::Plan=Plan(m, rng); kwargs...)::Dict{String,<:Any}
    # If we have a range, we just take a slice of the plan to enforce the range,
    # but taking into account the model max lag and max lead.
    plan = plan[rng,m];
    # We simulate as usual, but this time, with just plan properly adjusted
    # for the range
    return simulate(m, D1, plan; kwargs...)
end

# Simulate command with one date, just in case
function simulate(m::Model, D1::Dict{<:AbstractString,<:Any}, rng::MIT, plan::Plan=Plan(m, rng:rng); kwargs...)::Dict{String,<:Any}
    # return simulate(m,D1,rng:rng, plan; kwargs...)
    return simulate(m, D1, plan[rng:rng,m]; kwargs...)
end

import ..SimData

simulate(m::Model, D1::SimData, rng, plan::Plan=Plan(m, rng); kwargs...) = simulate(m, D1, plan[rng, m]; kwargs...)

function simulate(m::Model, D1::SimData, plan::Plan;
    deviation::Bool=false, overlay::Bool=false, kwargs...)::typeof(D1)
    ret = copy(D1)
    ig = get(kwargs, :initial_guess, zeros(0, 0))
    if ig isa SimData
        ig = ig[plan.range, m.varshks]
    end
    sim = simulate(m, plan, D1[plan.range, m.varshks]; kwargs..., initial_guess=ig)
    # overlay the sim data onto ret
    ret[plan.range, m.varshks] = sim
    return ret
end