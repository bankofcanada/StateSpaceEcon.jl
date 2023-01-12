##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


function shockdecomp(model::Model, p::Plan, exog_data::SimData;
    control::SimData=steadystatedata(model, p),
    deviation::Bool=false,
    anticipate::Bool=false,
    which::Symbol=model.options.which,
    verbose::Bool=getoption(model, :verbose, false),
    maxiter::Int=getoption(model, :maxiter, 20),
    tol::Float64=getoption(model, :tol, 1e-12),
    _debug=false
)

    if deviation
        error("`deviation` must be set to `false`.")
    end

    if anticipate
        error("`anticipate` must be set to `false`.")
    end

    # make sure we have first order solution
    sd = getsolverdata(model, :firstorder)::FirstOrderSD
    vm = sd.vm
    S = FOSimulatorData(plan, model, anticipate)

    # we will return result at the end
    result = Workspace()
    result.c = copy(control)

    # apply data transformations of @log variables
    exog_data = transform(exog_data[p.range, :], model)
    control = transform(control[p.range, :], model)

    result.sd = Workspace()
    let names = [:init, (v for (v, t) in vm.ex_vars if t == 0)...]
        for v in model.varshks
            push!(result.sd, v.name => MVTSeries(p.range, names, zeros))
        end
    end

    shocked = copy(exog_data)

    # running values at tnow
    # shock
    fill!(S.sol_t, 0)
    # control
    C_sol_t = copy(S.sol_t)
    # shock-decomp contributions (rows = endog, cols = exog)
    SD_t = zeros(S.nbck + S.nfwd, 1 + S.nex)

    # tnow is the Int index corresponding to the current period 
    tnow = model.maxlag
    # prepare initial conditions (only bck_t are used)
    for ind in S.ibck
        vind, tt = vm.inds_map[ind]
        vname = model.variables[vind].name
        S.sol_t[ind] = S1 = exog_data[tnow+tt, vind]
        C_sol_t[ind] = C1 = control[tnow+tt, vind]
        result.sd[vname][tnow+tt] = SD_t[ind, 1] = S1-C1
    end

    for tnow in model.maxlag+1:size(sim, 1)

        copyto!(S.bck_tm1, 1:S.nbck, S.sol_t, S.ibck)

        # prepare the right-hand-side of the system (that's the αₜ₋₁ part of the equation)
        # α_t .= sd.Zbb \ bck_t
        ldiv!(S.α_t, sd.Zbb, view(S.sol_t, S.ibck))
        # RHS .= sd.R * α_t
        # copyto!(S.RHS, sd.R * S.α_t)
        BLAS.gemv!('N', 1.0, sd.R, S.α_t, 0.0, S.RHS)


        # ldiv!()



    end



end