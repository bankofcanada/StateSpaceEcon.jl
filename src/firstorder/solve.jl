##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

import ModelBaseEcon.LittleDict

function solve!(model::Model)
    setsolverdata!(model, firstorder=FirstOrderSD(model))
    return model
end

struct VarMaps
    # map of model variables to their indexes in the model
    vi::LittleDict{Symbol,Int}
    nfwd::Int  # number of variables classified as forward looking 
    nbck::Int  # number of variables classified as backward looking 
    nex::Int   # number of variables classified as exogenous (includes shocks + @exog)
    oex::Int   # offset of exogenous class in the global indexing
    # maps of first-order variables by class : ind => (var.name, lag)
    bck_vars::Vector{Tuple{Symbol,Int}}
    fwd_vars::Vector{Tuple{Symbol,Int}}
    ex_vars::Vector{Tuple{Symbol,Int}}
    # inverse maps of first order variables by class: (var.name, lag) => ind
    bck_inds::LittleDict{Tuple{Symbol,Int},Int}
    #   note: indexing in fwd_inds continues from bck_inds
    fwd_inds::LittleDict{Tuple{Symbol,Int},Int}
    #   note: indexing in ex_inds restarts from 1
    ex_inds::LittleDict{Tuple{Symbol,Int},Int}
    # global map of first-order variables: fo-ind => (model var ind, lag)
    inds_map::Vector{Tuple{Int,Int}}
end

function VarMaps(model::Model)
    #
    # Precompute the index of each model variable 
    # Not related to first-order solver, just so we don't have to call indexin() all the time
    #
    vi = LittleDict{Symbol,Int}(
        var.name => ind for (ind, var) in enumerate(model.allvars)
    )
    #
    # define variables for lags and leads more than 1 
    # also categorize variables as fwd, bck, and ex
    #
    fwd_vars = Tuple{Symbol,Int}[]
    bck_vars = Tuple{Symbol,Int}[]
    ex_vars = Tuple{Symbol,Int}[]
    for mvar in model.allvars
        var = mvar.name
        lags, leads = extrema(lag for eqn in values(model.equations)
                              for (name, lag) in keys(eqn.tsrefs)
                              if name == var)
        if isexog(mvar) || isshock(mvar)
            push!(ex_vars, ((var, tt) for tt = lags:leads)...)
        elseif lags == 0 && leads == 0
            push!(bck_vars, (var, 0))
        else
            for tt = 1:-lags
                push!(bck_vars, (var, 1 - tt))
            end
            for tt = 1:leads
                push!(fwd_vars, (var, tt - 1))
            end
        end
    end
    #
    # build reverse indexing maps
    #
    nbck = length(bck_vars)
    nfwd = length(fwd_vars)
    nex = length(ex_vars)
    oex = nbck + nfwd
    #
    #
    bck_inds = LittleDict{Tuple{Symbol,Int},Int}(
        key => index for (index, key) in enumerate(bck_vars)
    )
    fwd_inds = LittleDict{Tuple{Symbol,Int},Int}(
        key => nbck + index for (index, key) in enumerate(fwd_vars)
    )
    ex_inds = LittleDict{Tuple{Symbol,Int},Int}(
        key => index for (index, key) in enumerate(ex_vars)
    )


    inds_map = Vector{Tuple{Int,Int}}(undef, nbck + nfwd + nex)
    for ((v, t), i) in bck_inds
        inds_map[i] = (vi[v], t)
    end
    for ((v, t), i) in fwd_inds
        inds_map[i] = (vi[v], t)
    end
    for ((v, t), i) in ex_inds
        inds_map[oex+i] = (vi[v], t)
    end

    return VarMaps(vi,
        nfwd, nbck, nex, oex,
        bck_vars, fwd_vars, ex_vars,
        bck_inds, fwd_inds, ex_inds,
        inds_map
    )
end

struct FirstOrderSystem
    FWD::Matrix{Float64}
    BCK::Matrix{Float64}
    EX::Matrix{Float64}
end

function FirstOrderSystem(JAC::SparseMatrixCSC, model::Model, vm::VarMaps=VarMaps(model))
    dim = vm.nbck + vm.nfwd
    sys = FirstOrderSystem(
        Matrix{Float64}(undef, dim, dim),
        Matrix{Float64}(undef, dim, dim),
        Matrix{Float64}(undef, dim, vm.nex),
    )
    fill_fosystem!(sys, JAC, model, vm)
end

function fill_fosystem!(sys::FirstOrderSystem, JAC::SparseMatrixCSC, model::Model, vm::VarMaps)
    # reset to 0
    FWD = fill!(sys.FWD, 0)
    BCK = fill!(sys.BCK, 0)
    EX = fill!(sys.EX, 0)
    #
    # build the system matrices
    # system:  FWD * x_t+1 + BCK * x_t + EX * e_t = 0
    #   where  x_t = [bck vars fwd vars]
    #          e_t = [exog vars]
    #
    fill!(FWD, 0.0)
    fill!(BCK, 0.0)
    fill!(EX, 0.0)
    for (eqind, col, val) in zip(findnz(JAC)...)
        # translate Jacobian column index `col` to variable index (in m.allvars) and time offset
        (vno, tt) = divrem(col - 1, 1 + model.maxlag + model.maxlead)
        vno += 1
        tt -= model.maxlag
        # obtain the variable name given its index
        var = model.allvars[vno].name
        var_tt = (var, tt)
        # is it in ex_vars
        ex_i = get(vm.ex_inds, var_tt, nothing)
        if ex_i !== nothing
            EX[eqind, ex_i] = val
            continue
        end
        # not in ex_vars. It's fwd_vars or bck_vars or both.
        if tt < 0
            # definitely bck_var
            bck_i = get(vm.bck_inds, (var, tt + 1), nothing)
            BCK[eqind, bck_i] = val
        elseif tt > 0
            # definitely fwd_var
            fwd_i = get(vm.fwd_inds, (var, tt - 1), nothing)
            FWD[eqind, fwd_i] = val
        else # tt == 0
            # could be either or both; 
            bck_i = get(vm.bck_inds, (var, 0), nothing)
            if bck_i !== nothing
                # prefer to treat it as bck_var, if both
                FWD[eqind, bck_i] = val
            else
                # not bck_fwd, must be fwd_fwd
                fwd_i = get(vm.fwd_inds, (var, 0), nothing)
                BCK[eqind, fwd_i] = val
            end
        end
        continue
    end
    # add links
    eqn = length(model.alleqns)
    for (var, tt) in vm.fwd_vars
        if tt == 0
            # add a fwd-bck cross link, if it's both fwd and bck variable
            b_i = get(vm.bck_inds, (var, 0), nothing)
            if b_i !== nothing
                eqn += 1
                f_i = vm.fwd_inds[(var, 0)]
                FWD[eqn, b_i] = 1
                BCK[eqn, f_i] = -1
            end
        else
            # add a fwd-fwd link
            eqn += 1
            FWD[eqn, vm.fwd_inds[(var, tt - 1)]] = 1
            BCK[eqn, vm.fwd_inds[(var, tt)]] = -1
        end
    end
    for (var, tt) in vm.bck_vars
        if tt == 0
            # cross links already done above
            continue
        else
            # add a fwd-fwd link
            eqn += 1
            FWD[eqn, vm.bck_inds[(var, tt)]] = 1
            BCK[eqn, vm.bck_inds[(var, tt + 1)]] = -1
        end
    end
    return sys
end

struct FirstOrderSD
    vm::VarMaps
    sys::FirstOrderSystem
    qz::QZData
    # various matrices 
    RbyZbb::Matrix{Float64}
    MAT::Matrix{Float64}
    # precomputed for empty plan
    MAT_n::Factorization
    MAT_x::SubArray{Float64,2,Matrix{Float64}}
end

function FirstOrderSD(model::Model)

    if !isempty(model.auxvars)
        error("Found auxiliary variables. Not yet implemented.")
    end

    if !islinearized(model)
        linearize!(model)
    end

    variant = :linearize
    refresh_med!(model, variant)
    lmed = getevaldata(model, variant)

    vm = VarMaps(model)
    sys = FirstOrderSystem(lmed.med.J, model, vm)

    # compute the Schur decomposition
    qz = run_qz(sys.FWD, sys.BCK, vm.nbck)
    # compute the system matrices
    RbyZbb, MAT = first_order_system(qz, sys.EX, vm.nbck, vm.nfwd, vm.nex)

    return FirstOrderSD(
        vm, sys, qz,
        RbyZbb, MAT,
        # pre-compute factorization for most common case (empty plan)
        lu(MAT[:, 1:(vm.oex)]),
        view(MAT, :, vm.oex .+ (1:vm.nex)))
end


function first_order_system(qz::QZData, EX::Matrix{Float64}, nbck::Int, nfwd::Int, nex::Int)
    oex = nbck + nfwd

    Zbb = lu(qz.Z[1:nbck, 1:nbck])
    Zbf = qz.Z[1:nbck, nbck.+(1:nfwd)]
    Zfb = qz.Z[nbck.+(1:nfwd), 1:nbck]
    Zff = qz.Z[nbck.+(1:nfwd), nbck.+(1:nfwd)]

    Tbb = qz.T[1:nbck, 1:nbck]
    Tbf = qz.T[1:nbck, nbck.+(1:nfwd)]
    # Tfb = qz.T[nbck.+(1:nfwd), 1:nbck]            # this one is 0
    Tff = lu(qz.T[nbck.+(1:nfwd), nbck.+(1:nfwd)])

    Sbb = qz.S[1:nbck, 1:nbck]
    # Sbf = qz.S[1:nbck, nbck.+(1:nfwd)]            # not needed
    # Sfb = qz.S[nbck.+(1:nfwd), 1:nbck]            # this one is 0
    # Sff = qz.S[nbck.+(1:nfwd), nbck.+(1:nfwd)]    # not needed

    QEX = qz.Q'EX

    R = vcat(-Tbb, Zff'Zfb)

    TiXf = Tff \ QEX[nbck.+(1:nfwd), :]

    MAT = zeros(nbck + nfwd, nbck + nfwd + nex)

    ### fill the backward-looking rows
    MAT[1:nbck, 1:nbck] = Sbb / Zbb
    # MAT[1:nbck, nbck .+(1:nfwd)] .= 0  # already zero 
    MAT[1:nbck, oex.+(1:nex)] = QEX[1:nbck, :] - (Tbf - Tbb * (Zbb \ Zbf)) * TiXf

    ### fill the forward-looking rows
    # MAT[nbck .+ (1:nfwd), 1:nbck] .= 0  # already zero 
    MAT[nbck.+(1:nfwd), nbck.+(1:nfwd)] = Zff'
    MAT[nbck.+(1:nfwd), oex.+(1:nex)] = TiXf

    return rdiv!(R, Zbb), MAT
end

