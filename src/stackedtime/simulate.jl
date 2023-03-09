##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

@inline function solve!(model::Model)
    return model
end

function sparse_solve(A, b)
    if A isa PardisoFactorization
        pardiso_solve!(A, copy(b))
    else
        A \ b
    end
end
"""
    sim_nr!(x, sd, maxiter, tol, verbose [, sparse_solver [, linesearch]])

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
  * `linesearch::Bool` - When `true` the Newton-Raphson is modified to include a 
    search along the descent direction for a sufficient decrease in f. It will 
    do this at each iteration. Default is `false`.

"""
function sim_nr!(x::AbstractArray{Float64}, sd::StackedTimeSolverData,
    maxiter::Int64, tol::Float64, verbose::Bool,
    sparse_solver::Function=(A, b) -> sparse_solve(A, b), linesearch::Bool=false)
    for it = 1:maxiter
        Fx, Jx = stackedtime_RJ(x, x, sd)
        nFx = norm(Fx, Inf)
        if nFx < tol
            if verbose
                @info "$it, || Fx || = $(nFx)"
            end
            return true
        end
        Δx = sparse_solver(Jx, Fx)
        λ = 1.0
        if linesearch
            nf = norm(Fx)
            # the Armijo rule: C.T.Kelly, Iterative Methods for Linear and Nonlinear Equations, ch.8.1, p.137
            α = 1e-4
            σ = 0.5
            while λ > 0.00001
                x_buf = copy(x)
                assign_update_step!(x_buf, -λ, Δx, sd)
                nrb2 = try
                    stackedtime_R!(Fx, x_buf, x_buf, sd)
                    norm(Fx)
                catch e
                    Inf
                end
                if nrb2 < (1.0 - α * λ) * nf
                    # if verbose && λ < 1.0
                    #     @info "Linesearch success with λ = $λ."
                    # end
                    break
                end
                λ = σ * λ
            end
            if verbose
                if λ <= 0.00001
                    @warn "Linesearch failed."
                elseif λ < 1.0
                    @info "Linesearch success with λ=$λ"
                end
            end
        end
        nΔx = λ * norm(vec(Δx), Inf)
        assign_update_step!(x, -λ, Δx, sd)
        if verbose
            @info "$it, || Fx || = $(nFx), || Δx || = $(nΔx)"
        end
        if nΔx < tol
            return true
        end
    end
    return false
end



function simulate(m::Model,
    p_ant::Plan,
    exog_ant::AbstractArray{Float64,2},
    p_unant::Plan=Plan(1U:0U, (;), falses(0, 0)),
    exog_unant::AbstractArray{Float64,2}=zeros(0, 0);
    #= Options =#
    anticipate::Bool=isempty(exog_unant),
    initial_guess::AbstractArray{Float64,2}=zeros(0, 0),
    #= Deviation options =#
    deviation::Bool=false,
    baseline::AbstractArray{Float64,2}=zeros(0, 0),
    deviation_ant=deviation,
    deviation_unant=deviation,
    #= Solver options =#
    variant::Symbol=m.options.variant,
    verbose::Bool=m.options.verbose,
    tol::Float64=m.options.tol,
    maxiter::Int=m.options.maxiter,
    fctype=getoption(m, :fctype, fcgiven),
    expectation_horizon::Union{Nothing,Int64}=nothing,
    #= Newton-Raphson options =#
    sparse_solver::Function=(A, b) -> sparse_solve(A, b),
    linesearch::Bool=getoption(m, :linesearch, false),
    warn_maxiter::Bool=getoption(getoption(m, :warn, Options()), :maxiter, false)
)

    unant_given = !isempty(exog_unant)

    if isempty(p_unant) == unant_given
        error("Invalid `unanticipated` inputs: either plan and data must both be given, or both must be left empty.")
    end

    if anticipate && unant_given
        error("Conflicting arguments: non-empty `exog_unanticipated` with `anticipate=true`.")
    end

    # make sure the model evaluation data is up to date
    refresh_med!(m, variant)

    NT = length(p_ant.range)
    nauxs = length(m.auxvars)
    nvarshks = length(m.varshks)
    logvars = [islog(var) | isneglog(var) for var in m.varshks]

    if size(exog_ant) != (NT, nvarshks)
        error("Incorrect dimensions of exog_data. Expected $((NT, nvarshks)), got $(size(exog_ant)).")
    end
    if !isempty(initial_guess) && size(initial_guess) != (NT, nvarshks)
        error("Incorrect dimensions of initial_guess. Expected $((NT, nvarshks)), got $(size(initial_guess)).")
    end

    if deviation_ant
        exog_ant = copy(exog_ant)
        if isempty(baseline)
            baseline = steadystatearray(m, p_ant)
        end
        if size(baseline) != (NT, nvarshks)
            error("Incorrect dimensions of baseline. Expected $((NT, nvarshks)), got $(size(baseline)).")
        end
        exog_ant[:, logvars] .*= baseline[:, logvars]
        exog_ant[:, .!logvars] .+= baseline[:, .!logvars]
    end
    exog_ant = ModelBaseEcon.update_auxvars(transform(exog_ant, m), m)

    if !isempty(initial_guess)
        x = ModelBaseEcon.update_auxvars(transform(initial_guess, m), m)
    else
        x = copy(exog_ant)
    end

    if anticipate
        gdata = StackedTimeSolverData(m, p_ant, fctype, variant)
        assign_exog_data!(x, exog_ant, gdata)
        if verbose
            @info "Simulating $(p_ant.range[1 + m.maxlag:NT - m.maxlead])" # anticipate gdata.FC
        end
        converged = sim_nr!(x, gdata, maxiter, tol, verbose, sparse_solver, linesearch)
        if warn_maxiter && !converged
            @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
        end
    else # unanticipated shocks

        #=== prepare sub-ranges ===#
        init = 1:m.maxlag
        term = NT .+ (1-m.maxlead:0)
        sim = 1+m.maxlag:NT-m.maxlead

        #=== prepare lists of indices according to types of variables ===#
        shkinds = findall(isshock, m.varshks)
        nshks = length(shkinds)

        varinds = findall(!isshock, m.varshks)
        nvars = length(varinds)

        # auxiliary vars are always last
        nvarshks = nvars + nshks
        varshkinds = 1:nvarshks

        nauxs = length(m.auxvars)
        auxinds = nvarshks .+ (1:nauxs)

        nallvars = nvarshks + nauxs
        allvarinds = 1:nallvars

        if unant_given
            #=== check compatibility of unanticipated inputs (data and plan) ===#
            if p_unant.range != p_ant.range
                error("Anticipated and unanticipated ranges don't match.")
            end
            if size(exog_unant) != size(exog_ant)
                error("Anticipated and unanticipated data  don't match.")
            end
            if deviation_unant
                exog_unant[:, logvars] .*= baseline[:, logvars]
                exog_unant[:, .!logvars] .+= baseline[:, .!logvars]
            end
            exog_unant = ModelBaseEcon.update_auxvars(transform(exog_unant, m), m)
        else
            #=== prepare unanticipated data and plan (backward compatibility) ===#
            p_unant = p_ant
            p_ant = Plan(m, p_unant.range[sim])
            exog_unant = copy(exog_ant)
            exog_ant[sim, shkinds] .= 0
            x[sim, shkinds] .= 0
        end

        x[init, allvarinds] .= exog_ant[init, allvarinds]
        t0 = first(sim)
        T = last(sim)
        if expectation_horizon === nothing
            # when expectation_horizon is not given, we simulate each iteration until the end and with the true final condition
            last_run = Workspace(; t=t0)
            for t in sim
                exog_inds = p_unant[t, Val(:inds)]
                psim = Plan(m, t:T)
                psim.exogenous .= p_ant.exogenous[begin+Int(t - t0):end, :]
                if t == t0
                    imaxiter = maxiter
                    itol = tol
                elseif (maximum(abs, x[t, exog_inds] .- exog_unant[t, exog_inds]) < tol) #= && (psim[t0, Val(:inds)] == exog_inds) =#
                    continue
                else
                    imaxiter = 5
                    itol = sqrt(tol)
                end
                setexog!(psim, t0, exog_inds)
                gdata = StackedTimeSolverData(m, psim, fctype, variant)
                x[t, exog_inds] = exog_unant[t, exog_inds]
                # assign_exog_data!(x[psim.range,:], exog_data[psim.range,:], gdata)
                sim_range = UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                assign_final_condition!(xx, exog_unant[sim_range, :], gdata)
                if verbose
                    @info "Simulating $(p_ant.range[t:T]) with $((itol, imaxiter))" # anticipate expectation_horizon gdata.FC
                end
                converged = sim_nr!(xx, gdata, imaxiter, itol, verbose, sparse_solver, linesearch)
                if warn_maxiter && !converged
                    @warn("Newton-Raphson reached maximum number of iterations (`imaxiter`).")
                end
                last_run = Workspace(; t, xx, gdata)
            end
            if last_run.t > t0
                local t = last_run.t
                xx = last_run.xx
                gdata = last_run.gdata
                if verbose
                    @info "Simulating $(p_ant.range[t:T]) with $((tol, maxiter))" # anticipate expectation_horizon gdata.FC
                end
                converged = sim_nr!(xx, gdata, maxiter, tol, verbose, sparse_solver, linesearch)
                if warn_maxiter && !converged
                    @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                end
            end
        else
            # when expectation_horizon is not given,
            # the first and last simulations use the true 
            # simulation range and final condition, while the intermediate 
            # simulations use expectation_horizon steps with fcnatural
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
                exog_inds = p_unant[t, Val(:inds)]
                psim = Plan(m, t:T)
                psim.exogenous .= p_ant.exogenous[begin+Int(t - t0):end, :]
                setexog!(psim, t0, exog_inds)
                sdata = StackedTimeSolverData(m, psim, fctype, variant)
                x[t, exog_inds] = exog_unant[t, exog_inds]
                sim_range = UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                assign_final_condition!(xx, exog_unant[sim_range, :], sdata)
                if verbose
                    @info "Simulating $(p_ant.range[t:T])" # anticipate expectation_horizon sdata.FC
                end
                converged = sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver, linesearch)
                if warn_maxiter && !converged
                    @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                end
            end
            # intermediate simulations
            last_t::Int64 = t0
            psim = Plan(m, 0:expectation_horizon-1)
            sdata = StackedTimeSolverData(m, psim, fcnatural, variant)
            for t in sim[2:end]
                exog_inds = p_unant[t, Val(:inds)]
                # we need to run a simulation if a variable is exogenous, or if a shock value is not zero
                # these intermediate simulations are always with fcnatural, 
                #       have length equal to expectation_horizon and 
                #       only the first period is imposed
                if (maximum(abs, x[t, exog_inds] .- exog_unant[t, exog_inds]) < tol) #= && (exog_inds == shkinds) =#
                    continue
                end
                psim1 = copy(psim)
                # the range of psim1 might extend beyond the range of p_ant.
                # we copy from p_ant as far as we have and copy the last line beyond that
                tmp_rng = t:min(t + expectation_horizon - 1, T)
                psim1.exogenous[t0.+(0:length(tmp_rng)-1), :] .= p_ant.exogenous[tmp_rng, :]

                # ===> must leave the psim1 plan empty beyond the end of p_ant
                # becasue we don't have data in exog_and for any exogenized
                # variables.
                # #=
                # for tt = length(tmp_rng)+1:expectation_horizon
                #     psim1.exogenous[t0+tt, :] .= p_ant.exogenous[T, :]
                # end
                # =#

                setexog!(psim1, t0, exog_inds)
                update_plan!(sdata, m, psim1)
                # note that the range always goes from 0 to expectation_horizon-1, 
                # so we need to add t in order to get the correct set of rows of x
                sim_range = t .+ UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                # The initial conditions are already set
                # The exogenous values are already set as well, except for the first period
                # In other words, we only need to impose the first period here
                xx[t0, exog_inds] = exog_unant[t, exog_inds]
                # Update the final conditions (the second argument is not used with fcnatural)
                assign_final_condition!(xx, zeros(0, nallvars), sdata)
                if verbose
                    @info("Simulating $(p_ant.range[t] .+ (0:expectation_horizon - 1))") # anticipate expectation_horizon sdata.FC
                end
                converged = sim_nr!(xx, sdata, 5, sqrt(tol), verbose, sparse_solver, linesearch)
                if warn_maxiter && !converged
                    @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                end
                last_t = t  # keep track of last simulation time
            end
            # last simulation
            if last_t > t0
                # do we need to re-run the last simulation?
                # if it didn't reach T, then yes
                # if the final condition is not fcnatural, then yes
                if (last_t + expectation_horizon != T) || (fctype != fcnatural)
                    psim = Plan(m, min(last_t + 1, T):T)
                    psim.exogenous .= p_ant.exogenous[end.+(1-length(psim.range):0), :]
                    # there are no unanticipated shocks in this simulation
                    sdata = StackedTimeSolverData(m, psim, fctype, variant)
                    # the initial conditions and the exogenous data are already in x
                    # we only need the final conditions
                    sim_range = UnitRange{Int}(psim.range)
                    xx = view(x, sim_range, :)
                    assign_final_condition!(xx, exog_unant[sim_range, :], sdata)
                    if verbose
                        @info "Simulating $(p_ant.range[last_t + 1:T])" # anticipate expectation_horizon sdata.FC
                    end
                    converged = sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver, linesearch)
                    if warn_maxiter && !converged
                        @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                    end
                end
            end
            # x = x[begin:end-expectation_horizon, :]
        end
    end

    x = x[axes(exog_ant)...]
    x .= inverse_transform(x, m)
    if deviation
        x[:, logvars] ./= baseline[:, logvars]
        x[:, .!logvars] .-= baseline[:, .!logvars]
    end

    return x
end


