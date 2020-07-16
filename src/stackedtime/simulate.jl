
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
                sparse_solver::Function = (A, b)->A \ b)
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


"""
function simulate(m::Model, p::Plan, exog_data::AbstractArray{Float64,2};
                    initial_guess::AbstractArray{Float64,2} = zeros(0, 0),
                    linearize::Bool = false,
                    deviation::Bool = false,
                    anticipate::Bool = true,
                    verbose::Bool = m.options.verbose,
                    tol::Float64 = m.options.tol,
                    maxiter::Int64 = m.options.maxiter,
                    fctype::FCType = fcgiven,
                    expectation_horizon::Union{Nothing,Int64} = nothing,
                    sparse_solver::Function = (A, b)->A \ b
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
        linearize!(m; deviation = deviation)
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


