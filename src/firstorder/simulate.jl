##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

import ModelBaseEcon.FirstOrderMED

# specialization of simulate() for first-order solution

function simulate(model::Model, plan::Plan, exog::AbstractMatrix;
    deviation::Bool=false,
    anticipate::Bool=false,
    baseline::AbstractMatrix{Float64}=zeros(0, 0),
    verbose::Bool=model.options.verbose
)

    if anticipate
        error("`anticipate=true` not yet implemented")
    end

    # make sure we have first order solution
    if !isfirstorder(model) || !(model.solverdata isa FirstOrderSD)
        error("First-order solution is not ready. Call `solve!(m, :firstorder)`")
    end

    if !isempty(model.auxvars)
        error("Found auxiliary variables. Not yet implemented.")
    end

    #= 
    #TODO: what about log vars ?!
    if any(islog, model.varshks) || any(isneglog, model.varshks)
        error("Found log/neglog vars. Not yet implemented.")
    end
    =#

    # transform exogenous data 
    logvars = [islog(var) | isneglog(var) for var in model.varshks]
    need_trans = any(logvars)
    need_baseline = !deviation || (deviation && need_trans)
    if isempty(baseline) && need_baseline
        baseline = steadystatearray(model, plan)
    end

    if need_trans && deviation
        exog = copy(exog)
        exog[:, logvars] .*= baseline[:, logvars]
        exog[:, .!logvars] .+= baseline[:, .!logvars]
    end
    if need_trans
        exog = transform(exog, model)
        baseline_tr = transform(baseline, model)
    else
        exog = copy(exog)
        baseline_tr = baseline
    end

    # we work in deviation from steady state
    if !deviation
        # already transformed, so just subtract
        exog .-= baseline_tr
    end

    ed = model.evaldata::FirstOrderMED
    sd = model.solverdata::FirstOrderSD
    vi = LittleDict{Symbol,Int}(
        var.name => ind for (ind, var) in enumerate(model.allvars)
    )

    nbck = length(ed.bck_vars)
    nfwd = length(ed.fwd_vars)
    nex = length(ed.ex_vars)

    # running solution vector
    sol_t = Vector{Float64}(undef, nbck + nfwd + nex)

    # indexes of the different types of variables within sol_t
    ibck = 1:nbck
    bck_t = view(sol_t, ibck)
    ifwd = nbck .+ (1:nfwd)
    fwd_t = view(sol_t, ifwd)
    ex_offset = nbck + nfwd
    iex = ex_offset .+ (1:nex)
    ex_t = view(sol_t, iex)

    # the transformed bck_t
    α_t = similar(bck_t)

    # exogenous flags (true = exogenous, false = endogenous)
    xflags_t = similar(sol_t, Bool)

    # the result will be saved in sim
    sim = copy(exog)

    ##################
    # First period
    # Let’s start at the very beginning, a very good place to start.
    ##################

    # tnow is the row-index in exog corresponding to the current period 
    tnow = model.maxlag

    # prepare initial conditions (only bck_t are used)
    let ind = 1
        while ind <= nbck
            vind, tt = sd.inds_map[ind]
            bck_t[ind] = exog[tnow+tt, vind]
            ind += 1
        end
    end

    ##################
    # The loop
    ##################

    RHS = Vector{Float64}(undef, nbck + nfwd)
    for tnow in model.maxlag+1:size(sim, 1)

        # prepare the right-hand-side of the system (that's the αₜ₋₁ part of the equation)
        # α_t .= sd.Zbb \ bck_t
        ldiv!(α_t, sd.Zbb, bck_t)
        # RHS .= sd.R * α_t
        copyto!(RHS, sd.R * α_t)

        # prepare the exogenous mask vector xflags_t
        # default is all bck & fwd are endo and all ex are exog
        fill!(xflags_t, false)
        xflags_t[iex] .= true
        for (solind, (varind, tt)) in zip(iex, sd.inds_map[iex])
            sol_t[solind] = exog[tnow+tt, varind]
        end
        # modify according to plan (only contemporaneous)
        empty_plan = true
        for (var, xflag) in zip(model.varshks, plan.exogenous[tnow, :])
            vname = var.name
            solind = get(ed.bck_inds, (vname, 0), -1)
            if solind > -1
                xflags_t[solind] = xflag
                if xflag
                    empty_plan = false
                    sol_t[solind] = exog[tnow, vi[vname]]
                end
                continue
                # it's possible for (var,0) to be both in bck and fwd (mixed variable)
                # if exogenous, we set only one of them, not both
                # we flipped a coin and bck won the honour of being preferred
            end
            solind = get(ed.fwd_inds, (vname, 0), -1)
            if solind > -1
                xflags_t[solind] = xflag
                if xflag
                    empty_plan = false
                    sol_t[solind] = exog[tnow, vi[vname]]
                end
                continue
            end
            solind = get(ed.ex_inds, (vname, 0), -1)
            if solind > -1
                xflags_t[ex_offset+solind] = xflag
                if !xflag
                    empty_plan = false
                end
                continue
            end
            error("Variable $(vname)[t] not found in solution vector!?!?!")
        end

        @assert length(xflags_t) - sum(xflags_t) == nbck + nfwd "Incorrect number of endogenous unknowns at $(plan.range[tnow])."

        # solve for the endogenous
        if empty_plan
            # MAT_n is already LU-decomposed, so this is branch should be faster
            sol_t[.!xflags_t] = sd.MAT_n \ (RHS - sd.MAT_x * sol_t[xflags_t])
        else
            sol_t[.!xflags_t] = sd.MAT[:, .!xflags_t] \ (RHS - sd.MAT[:, xflags_t] * sol_t[xflags_t])
        end

        # TODO: This mapping can be precomputed!  Populate the sim
        for (simind, var) in enumerate(model.varshks)
            vname = var.name
            solind = get(ed.bck_inds, (vname, 0), -1)
            if solind > -1
                sim[tnow, simind] = sol_t[solind]
                continue
            end
            solind = get(ed.fwd_inds, (vname, 0), -1)
            if solind > -1
                sim[tnow, simind] = sol_t[solind]
                continue
            end
            solind = get(ed.ex_inds, (vname, 0), -1)
            if solind > -1
                sim[tnow, simind] = sol_t[ex_offset+solind]
                continue
            end
            error("Variable $(vname)[t] not found in solution vector!?!?!")
        end

    end

    # inverse transform data to give back to the user

    # sim is in deviation
    if !deviation
        sim .+= baseline_tr
    end
    if need_trans
        sim = inverse_transform(sim, model)
    end

    return sim
end

