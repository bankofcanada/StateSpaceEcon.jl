##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# version of simulate for stochastic simulations 

#  NOTE: This requires TimeSeriesEcon v0.5.1 where map(func, ::Workspace) returns a Workspace.
_make_result_container(shocks::Workspace, basedata)::Workspace = map(_ -> copy(basedata), shocks)
_make_result_container(shocks::Vector, basedata)::Vector{MaybeSimData} = convert(Vector{MaybeSimData}, map(_ -> copy(basedata), shocks))

function stoch_simulate(m::Model, p::Plan, baseline::SimData, shocks;
    check::Bool=false,
    #= Solver options =#
    variant::Symbol=m.options.variant,
    verbose::Bool=m.options.verbose,
    tol::Float64=m.options.tol,
    maxiter::Int=m.options.maxiter,
    fctype=getoption(m, :fctype, fcgiven),
    #= Newton-Raphson options =#
    linesearch::Bool=getoption(m, :linesearch, false),
    warn_maxiter=getoption(getoption(m, :warn, Options()), :maxiter, false),
    sim_solver=:sim_nr
)

    sim_solve! =
        sim_solver == :sim_nr ? sim_nr! :
        sim_solver == :sim_lm ? sim_lm! :
        sim_solver == :sim_gn ? sim_gn! :
        error("Unknown solver $sim_solver.")

    if isempty(shocks)
        return _make_result_container(shocks, baseline)
    end

    # get the range of all shocks realizations (this will be the full simulation range)
    shkrng = mapreduce(rangeof, union, values(shocks))

    # are all shocks in the same range
    same_range = all(==(shkrng) ∘ rangeof, values(shocks))

    # make sure the plan range contains shkrng 
    if firstdate(p) + m.maxlag > first(shkrng)
        throw(ArgumentError("Simulation starts too late for the given shocks realizations."))
    end
    if last(shkrng) > lastdate(p) - m.maxlead
        throw(ArgumentError("Simulation ends too early for the given shocks realizations."))
    end
    if last(shkrng) > lastdate(p) - m.maxlead - 20  # why 20? 
        @warn "Simulation may be too short - allow at least 20 periods after the last shock."
    end

    # make sure all given stochastic shocks are exogenous during their stochastic ranges
    for (key, value) in pairs(shocks)
        for (shk, val) in pairs(value)
            tinds = Plans._offset(p, rangeof(val))
            vind = p.varshks[shk]
            if !all(p.exogenous[tinds, vind])
                @warn "$shk in shocks[$key] is endogenous in the given plan."
            end
        end
    end

    # make sure the model evaluation data is up to date
    refresh_med!(m, variant)

    # range for the result
    resrng = first(shkrng)-m.maxlag:last(p.range)

    # prepare the baseline data
    e₀ = ModelBaseEcon.update_auxvars(transform(baseline[resrng, m.varshks], m), m)

    if check
        ### Anticipated run
        # the plan
        local p₀ = copy(p[resrng])
        # the solver data
        local d₀ = StackedTimeSolverData(m, p₀, fctype, variant)

        # we assume that baseline is the anticipated solution. let's check 
        local res = Vector{Float64}(undef, size(d₀.J, 1))
        stackedtime_R!(res, e₀, e₀, d₀)
        local nres = norm(res, Inf)
        if nres >= tol
            throw(ArgumentError("The given baseline is not a solution: residual $nres > tolerance $tol."))
        end
    end

    ### Unanticipated stochastic shocks

    # allocate results 
    results = _make_result_container(shocks, e₀)

    # last simulation period, excluding final conditions periods
    sim_end = lastdate(p) - m.maxlead

    # the time loop
    for i = eachindex(shkrng)
        t = shkrng[i]

        # simulation range for this period
        # rₜ = t:sim_end

        # simulation plan is a view into the full plan
        pₜ = let
            i1 = Plans._offset(p, t) - m.maxlag
            i2 = lastindex(p.exogenous, 1)
            Plan{typeof(t)}(t-m.maxlag:lastdate(p), p.varshks, view(p.exogenous, i1:i2, :))
        end

        # solver data 
        dₜ = StackedTimeSolverData(m, pₜ, fctype, variant)

        # the shocks realizations loop
        for ((skey, shock), (rkey, result)) in zip(pairs(shocks), pairs(results))
            @assert skey == rkey

            # skip if this simulation has already failed
            isfailed(result) && continue

            # skip if t is outside the range of shock
            same_range || (firstdate(shock) ≤ t ≤ lastdate(shock)) || continue

            verbose && @info "Simulating $skey over $(t:sim_end)."

            # create a view into the result for this period
            eₜ = view(result, pₜ.range, :)

            # assign unanticipated shocks
            eₜ[t, axes(shock, 2)] .+= shock[t, :]   # 

            # solve 
            try
                converged = sim_solve!(eₜ, dₜ, maxiter, tol, verbose, linesearch)
                check_converged(converged, warn_maxiter)
            catch
                # marked as failed 
                results[rkey] = SimFailed(t)
                continue
            end

        end # shocks loop 

    end # time loop

    # strip auxvar columns and inverse transform
    have_auxs = (m.nauxs > 0)
    for (key, result) in pairs(results)
        isfailed(result) && continue
        if have_auxs
            result = result[:, m.varshks]
        end
        results[key] = inverse_transform(result, m)
    end

    return results
end


