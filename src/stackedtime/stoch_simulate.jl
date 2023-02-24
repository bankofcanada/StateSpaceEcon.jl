##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# version of simulate for stochastic simulations 


function _make_result_container(shocks::Workspace, basedata)
    w = Workspace()
    for key in keys(shocks)
        push!(w, key => copy(basedata))
    end
    return w
end

function _make_result_container(shocks::Vector, basedata)
    return map(copy, Iterators.repeated(basedata, length(shocks)))
end

function stoch_simulate(m::Model, p::Plan, baseline::SimData, shocks;
    check::Bool=false,
    #= Solver options =#
    variant::Symbol=m.options.variant,
    verbose::Bool=m.options.verbose,
    tol::Float64=m.options.tol,
    maxiter::Int=m.options.maxiter,
    fctype=getoption(m, :fctype, fcgiven),
    #= Newton-Raphson options =#
    sparse_solver::Function=\,
    linesearch::Bool=getoption(m, :linesearch, false),
    warn_maxiter::Bool=getoption(getoption(m, :warn, Options()), :maxiter, false)
)

    # get the range of all shocks realizations (this will be the full simulation range)
    shkrng = mapreduce(rangeof, union, values(shocks))

    # are all shocks in the same range
    same_range = all(==(shkrng) ∘ rangeof, shocks)

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
                throw(ArgumentError("$shk in shocks realization $key is endogenous in the given plan."))
            end
        end
    end

    # make sure the model evaluation data is up to date
    refresh_med!(m, variant)

    # prepare the baseline data
    e₀ = ModelBaseEcon.update_auxvars(transform(baseline, m), m)

    if check
        ### Anticipated run
        # the plan
        local p₀ = copy(p)
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

    # range for the result
    resrng = first(shkrng)-m.maxlag:last(p.range)

    # allocate results 
    results = _make_result_container(shocks, e₀[resrng, m.varshks])

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

            # skip if t is outside the range of shock
            same_range || firstdate(shock) ≤ t ≤ lastdate(shock) || continue

            verbose && @info "Simulating $skey over $(t:sim_end)."

            # create a view into the result for this period
            eₜ = view(result, pₜ.range, :)

            # assign unanticipated shocks
            eₜ[t, axes(shock, 2)] .+= shock[t, :]   # 

            # solve 
            converged = sim_nr!(eₜ, dₜ, maxiter, tol, verbose, sparse_solver, linesearch)
            if warn_maxiter && !converged
                @warn("Newton-Raphson reached maximum number of iterations for $skey at $(t:sim_end)")
            end

        end # shocks loop 

    end # time loop

    # strip auxvar columns and inverse transform
    for (key, result) in pairs(results)
        if m.nauxs > 0
            result = result[:, m.varshks]
        end
        results[key] = inverse_transform(result, m)
    end

    return results
end


