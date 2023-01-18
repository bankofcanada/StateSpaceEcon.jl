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
    variant::Symbol=model.options.variant,
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

    # preallocate the shockdecomp MVTSeries
    result.sd = Workspace()
    let colnames = [:init, (v for (v, t) in vm.ex_vars if t == 0)..., :nonlinear]
        for v in model.varshks
            if !(isshock(v) || isexog(v))
                push!(result.sd, v.name => MVTSeries(p.range, colnames, zeros))
            end
            continue
        end
    end
    shocked = copy(exog_data)

    # running values at tnow
    # shock
    fill!(S.sol_t, 0)
    # shock-decomp contributions (rows = endog, cols = exog)
    nrhs = 1 + S.nex
    SD_t = zeros(S.nbck + S.nfwd, nrhs)
    SD_RHS = similar(SD_t)
    SD_EX = zeros(S.nex, S.nex)

    # tnow is the Int index corresponding to the current period 
    tnow = model.maxlag
    # prepare initial conditions (only bck_t are used)
    for ind in S.ibck
        vind, tt = vm.inds_map[ind]
        vname = model.variables[vind].name
        S.sol_t[ind] = S1 = exog_data[tnow+tt, vind]
        C1 = control[tnow+tt, vind]
        result.sd[vname][tnow+tt] = SD_t[ind, 1] = S1 - C1
    end

    # pre-compute the matrix of t-1
    RbyZbb = copy(sd.R)
    rdiv!(RbyZbb, sd.Zbb)

    for tnow in model.maxlag+1:size(shocked, 1)

        # RHS related to t-1
        # for shocked
        BLAS.gemv!('N', 1.0, RbyZbb, view(S.sol_t, S.ibck), 0.0, S.RHS)
        # for contributions
        BLAS.gemm!('N', 'N', 1.0, RbyZbb, view(SD_t, S.ibck, :), 0.0, SD_RHS)


        # fill exogenous data
        for (ind0, ind, (varind, tt)) in zip(1:S.nex, S.iex, vm.inds_map[S.iex])
            # for shocked
            S.sol_t[ind] = S1 = exog_data[tnow+tt, varind]
            # for contributions
            C1 = control[tnow+tt, varind]        #= C_sol_t[ind] = =#
            SD_EX[ind0, ind0] = S1 - C1
        end

        # RHS related to exogenous
        # for shocked
        BLAS.gemv!('N', -1.0, sd.MAT_x, view(S.sol_t, S.iex), 1.0, S.RHS)
        # for contributions
        BLAS.gemm!('N', 'N', -1.0, sd.MAT_x, SD_EX, 1.0, view(SD_RHS, :, 2:nrhs))

        # solve 
        # for shocked
        ldiv!(view(S.sol_t, S.ien), sd.MAT_n, S.RHS)
        # for contributions
        ldiv!(SD_t, sd.MAT_n, SD_RHS)

        # assign values where they belong (only unique ones)
        for uniq_ind in S.uniq_inds_map
            uniq_ind > S.oex && continue
            vind, tlag = vm.inds_map[uniq_ind]
            tlag == 0 || continue
            var = model.allvars[vind]
            shocked[tnow, vind] = S.sol_t[uniq_ind]
            result.sd[var.name][begin+tnow-1, :init] = SD_t[uniq_ind, 1]
            for (sind, (sname, _)) in enumerate(vm.ex_vars)
                result.sd[var.name][sname][tnow] += SD_t[uniq_ind, 1+sind]
            end
        end

    end

    # update nonlinear
    for (var, data) in pairs(result.sd)
        data.nonlinear .= shocked[var] - control[var] - sum(data, dims=2)
    end

    # assign shocked to the result
    result.s = copy(result.c)
    result.s[p.range, :] .= shocked

    return result
end