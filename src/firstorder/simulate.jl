##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

# specialization of simulate() for first-order solution

"""
Find the last period in which the plan is non-empty.

Reminder: an empty plan is one in which all variables are endogenous and all
shocks are exogenous, i.e., no swapping has been done.
"""
function _last_swapped_period(plan::Plan, model::Model)
    shks = Bool[isshock(v) || isexog(v) for v in model.varshks]
    vars = .!shks
    simrng = model.maxlag+1:length(plan.range)-model.maxlead
    for per in reverse(simrng)
        is_empty = all(plan.exogenous[per, shks]) && !any(plan.exogenous[per, vars])
        if !is_empty
            return per
        end
    end
    return -1
end

mutable struct FOSimulatorData{A}
    t_last_swap::Int
    empty_plan::Bool
    nbck::Int
    nfwd::Int
    nex::Int
    oex::Int
    ibck::UnitRange{Int}
    ifwd::UnitRange{Int}
    ien::UnitRange{Int}
    iex::UnitRange{Int}
    sol_t::Vector{Float64}
    α_t::Vector{Float64}
    bck_tm1::Vector{Float64}
    sol_tp1::Vector{Float64}
    RHS::Vector{Float64}
    xflags_t::Vector{Bool}
    varmaxlead::Vector{Int}
    uniq_inds_map::Vector{Int}
end
function FOSimulatorData(plan::Plan, model::Model, anticipate::Bool)
    t_last_swap = _last_swapped_period(plan, model)

    sd = getsolverdata(model, :firstorder)::FirstOrderSD
    vm = sd.vm

    nbck = vm.nbck
    nfwd = vm.nfwd
    nex = vm.nex
    oex = vm.oex

    varmaxlead = zeros(Int, model.nvarshks)
    for (vind, tt) in vm.inds_map[nbck.+(1:nfwd)]
        if tt > varmaxlead[vind]
            varmaxlead[vind] = tt
        end
    end

    return FOSimulatorData{anticipate}(t_last_swap, t_last_swap <= model.maxlag,
        nbck, nfwd, nex, oex,
        1:nbck, nbck .+ (1:nfwd), 1:oex, oex .+ (1:nex), # ibck, ifwd, ien, iex
        Vector{Float64}(undef, nbck + nfwd + nex), # sol_t
        Vector{Float64}(undef, nbck), # α_t
        Vector{Float64}(undef, nbck), # bck_tm1
        Vector{Float64}(undef, nbck + nfwd + nex), # sol_tp1
        Vector{Float64}(undef, nbck + nfwd), # RHS
        Vector{Bool}(undef, nbck + nfwd + nex), # xflags_t
        varmaxlead,
        indexin(unique(vm.inds_map), vm.inds_map),
    )
end

function simulate(model::Model, plan::Plan, exog::AbstractMatrix;
    deviation::Bool=false,
    anticipate::Bool=false,
    baseline::AbstractMatrix{Float64}=zeros(0, 0),
    verbose::Bool=model.options.verbose,
    #= nlcorrect::Bool=false, =#
    kwargs...
)

    if !isempty(model.auxvars)
        error("Found auxiliary variables. First-order solver not yet implemented for models with auxiliary variables.")
    end

    # make sure we have first order solution
    sd = getsolverdata(model, :firstorder)::FirstOrderSD
    vm = sd.vm
    S = FOSimulatorData(plan, model, anticipate)

    if anticipate && !S.empty_plan
        @warn "Running linearized stacked-time solver."
        return simulate(model, plan, exog;
            deviation, anticipate, verbose, baseline, kwargs...)
    end

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

    # the result will be saved in sim
    sim = copy(exog)

    ##################
    # First period
    # Let’s start at the very beginning, a very good place to start.
    ##################

    # tnow is the row-index in exog corresponding to the current period 
    tnow = model.maxlag

    # prepare initial conditions (only bck_t are used)
    for ind in S.ibck
        vind, tt = vm.inds_map[ind]
        S.sol_t[ind] = exog[tnow+tt, vind]
    end

    ##################
    # The loop
    ##################

    for tnow in model.maxlag+1:size(sim, 1)

        copyto!(S.bck_tm1, 1:S.nbck, S.sol_t, S.ibck)

        # prepare the right-hand-side of the system (that's the αₜ₋₁ part of the equation)
        # α_t .= sd.Zbb \ bck_t
        ldiv!(S.α_t, sd.Zbb, view(S.sol_t, S.ibck))
        # RHS .= sd.R * α_t
        # copyto!(S.RHS, sd.R * S.α_t)
        BLAS.gemv!('N', 1.0, sd.R, S.α_t, 0.0, S.RHS)

        # fill exogenous data
        for (ind, (varind, tt)) in zip(S.iex, vm.inds_map[S.iex])
            S.sol_t[ind] = exog[tnow+tt, varind]
        end

        dispatch = tnow > S.t_last_swap ? Val(:empty_plan) : Val(:swapped_plan)

        fo_sim_step!(tnow, S, sd,
            model, plan, exog,
            dispatch)

        #= if nlcorrect && tnow + model.maxlead <= size(sim, 1)
            fo_nl_correct!(tnow, S, sd, ed,
                model, plan, exog,
                baseline_tr[tnow-model.maxlag:tnow+model.maxlead, :],
                dispatch)
        end =#

        # TODO: This mapping can be precomputed!  
        # Populate the sim
        for (simind, var) in enumerate(model.varshks)
            vname = var.name
            solind = get(vm.bck_inds, (vname, 0), -1)
            if solind > -1
                sim[tnow, simind] = S.sol_t[solind]
                continue
            end
            solind = get(vm.fwd_inds, (vname, 0), -1)
            if solind > -1
                sim[tnow, simind] = S.sol_t[solind]
                continue
            end
            solind = get(vm.ex_inds, (vname, 0), -1)
            if solind > -1
                sim[tnow, simind] = S.sol_t[S.oex+solind]
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

#############################
#  fo_sim_step!
#############################

function fo_sim_step!(
    tnow::Int,
    S::FOSimulatorData,
    sd::FirstOrderSD,
    model::Model,
    plan::Plan,
    exog::AbstractMatrix{Float64},
    ::Val{:empty_plan}
)
    fill!(S.xflags_t, false)
    S.xflags_t[S.iex] .= true
    S.sol_t[S.ien] = sd.MAT_n \ (S.RHS - sd.MAT_x * S.sol_t[S.iex])
    return nothing
end

function fo_sim_step!(
    tnow::Int,
    S::FOSimulatorData,
    sd::FirstOrderSD,
    model::Model,
    plan::Plan,
    exog::AbstractMatrix{Float64},
    ::Val{:swapped_plan}
)
    vm = sd.vm
    # prepare exogenous flags and data according to plan
    fill!(S.xflags_t, false)
    S.xflags_t[S.iex] .= true
    empty_plan = true  # check if the plan is really empty
    for (var, xflag) in zip(model.varshks, plan.exogenous[tnow, :])
        vname = var.name
        ind = -1
        for (inds, def_xflag, offset) in ((vm.bck_inds, false, 0),
            (vm.fwd_inds, false, 0),
            (vm.ex_inds, true, S.oex))
            # def_xflag : default xflag in empty plan, i.e. `false` for variables, `true` for shocks 
            # Note: bck_inds listed first for a good reason
            # it's possible for (var,0) to be both in bck and fwd (mixed variable)
            # if exogenous, we set only one of them, not both
            # we flipped a coin and bck won the honour of being preferred
            ind = get(inds, (vname, 0), -1)
            if ind > -1
                oind = offset + ind
                S.xflags_t[oind] = xflag
                if xflag
                    S.sol_t[oind] = exog[tnow, vm.vi[vname]]
                end
                if xflag != def_xflag
                    empty_plan = false
                end
                break
            end
        end
        if ind < 0
            error("Variable not found in plan: $vname")
        end
    end
    # solve for the endogenous unknowns
    if empty_plan
        # MAT_n is already LU-factorized, so this branch should be faster
        S.sol_t[S.ien] = sd.MAT_n \ (S.RHS - sd.MAT_x * S.sol_t[S.iex])
    else
        # check plan
        if sum(S.xflags_t) != S.nex
            error("Incorrect number of endogenous unknowns in plan at $(plan.range[tnow]).")
        end
        S.sol_t[.!S.xflags_t] = sd.MAT[:, .!S.xflags_t] \ (S.RHS - sd.MAT[:, S.xflags_t] * S.sol_t[S.xflags_t])
    end
    return nothing
end

#############################
#  fo_nl_correct!
#############################

#= 

##### Ignore. This is a failed experiment - might come back to it later.

function update_tpoint!(tpoint, tzero, sd, S, check=true, tol=eps() * 1024)
    update = fill!(similar(tpoint), 0.0)
    # t-1
    for ((vind, tt), val) in zip(sd.inds_map[S.ibck], S.bck_tm1)
        update[tzero-1+tt, vind] = val
    end
    # t
    for ((vind, tt), val) in zip(sd.inds_map, S.sol_t)
        if tt < 0
            if check
                val1 = update[tzero+tt, vind]
                @assert ≈(val1, val, atol=tol, rtol=sqrt(tol) / 128)
            end
        else
            update[tzero+tt, vind] = val
        end
    end
    # t+1
    for ((vind, tt), val) in zip(sd.inds_map, S.sol_tp1)
        if tt < S.varmaxlead[vind]
            if check
                val1 = update[tzero+1+tt, vind]
                @assert ≈(val1, val, atol=tol, rtol=sqrt(tol) / 128)
            end
        else
            update[tzero+1+tt, vind] = val
        end
    end
    tpoint .+= update
    return tpoint
end


function fo_nl_correct!(tnow::Int,
    S::FOSimulatorData,
    sd::FirstOrderSD,
    ed::FirstOrderMED,
    model::Model,
    plan::Plan,
    exog::AbstractMatrix{Float64},
    baseline::AbstractMatrix{Float64},
    ::Val{:empty_plan};
    tol::Float64=getoption(model, :tol, eps() * 1024),
    maxiter::Int=getoption(model, :maxiter, 5))

    tpoint = copy(baseline)
    tzero = model.maxlag + 1
    # S.fwd_tp1 .= sd.ZfbByZbb * S.sol_t[1:S.nbck]
    # copyto!(S.fwd_tp1, sd.ZfbByZbb * S.sol_t[1:S.nbck])
    # BLAS.gemv!('N', 1.0, sd.ZfbByZbb, S.sol_t[1:S.nbck], 0.0, S.fwd_tp1)
    S.sol_tp1[S.iex] .= 0
    S.sol_tp1[S.ien] = sd.MAT_n \ (sd.R * (sd.Zbb \ S.sol_t[S.ibck]))
    update_tpoint!(tpoint, tzero, sd, S)

    S1 = deepcopy(S)
    fill!(S1.bck_tm1, 0.0)
    fill!(S1.sol_t, 0.0)
    fill!(S1.sol_tp1, 0.0)
    fill!(S1.RHS, 0.0)
    qz1 = deepcopy(sd.qz)

    med = ModelBaseEcon.ModelEvaluationData(model)
    RES, JAC = eval_RJ(tpoint, med)
    nres = length(RES)

    # we need the endogenous indexes
    # the mask S.xflags_t won't work becasue MAT is a different 2-nd dimension
    #   (because we're appending the residual "shocks")
    ninds = findall(!, S.xflags_t)
    # indexes corresponding to residual "shocks" 
    #   ( they are added at the very end, beyond the regular MAT columns)
    rinds = size(sd.MAT, 2) .+ (1:nres)

    EX0 = similar(ed.EX)

    iter = 0
    while norm(RES) > tol && iter <= maxiter
        iter += 1

        # prepare FWD, BCK and EX given JAC
        ModelBaseEcon.fill_fo_matrices!(qz1.S, qz1.T, EX0, JAC, model, ed)
        # compute the Schur decomposition
        _run_qz!(qz1, S.nbck)
        # append the columns for the residual to the EX matrix
        EX1 = hcat(EX0, diagm(S.nbck + S.nfwd, nres, RES))
        # build the system matrix 
        Zbb, ZfbByZbb, R, MAT = first_order_system(qz1, EX1, S.nbck, S.nfwd, S.nex + nres)

        MAT_n = lu(MAT[:, ninds])
        RESSHK = sum(MAT[:, rinds], dims=2)

        # Solve for the endogenous unknowns
        S1.sol_t[ninds] = -(MAT_n \ RESSHK)
        # NOTE 1: It should be MAT[:, rinds] * ones(nres), but this is the sum the columns of MAT[:,rinds]
        # NOTE 2: the contribution of alpha_t-1 to the RHS is 0.
        # Also, the regular exogenous columns of MAT are multiplied by 0.
        # We simply don't include these two, and we include only the contribution
        #      from the residual.

        if norm(S1.sol_t) < tol
            break
        end

        # Solve for fwd unknowns one period forward
        S1.sol_tp1[S.iex] .= 0
        S1.sol_tp1[S.ien] = MAT_n \ (R * (Zbb \ S1.sol_t[S.ibck]) - 0RESSHK)
        # BLAS.gemv!('N', 1.0, ZfbByZbb, S1.sol_t[1:S.nbck], 0.0, S1.fwd_tp1)

        # update the solution 
        S.sol_t .+= S1.sol_t

        # update tpoint
        update_tpoint!(tpoint, tzero, sd, S1)

        RES, JAC = eval_RJ(tpoint, med)
    end

    return

end

=#

