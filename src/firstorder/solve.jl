##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

import ModelBaseEcon.FirstOrderMED
import ModelBaseEcon.LittleDict

function solve!(model::Model)
    firstorder!(model)
    setsolverdata!(model, firstorder = FirstOrderSD(model))
    return model
end

struct FirstOrderSD
    qz::QZData
    vi::LittleDict{Symbol,Int}
    # translations from FO variable index to Var-Lag pair
    inds_map::Vector{Tuple{Int,Int}}
    # steady state (or point of linearization)
    ss_vec::Vector{Float64}
    # various matrices 
    Zbb::Factorization
    ZfbByZbb::Matrix{Float64}
    R::Matrix{Float64}
    MAT::Matrix{Float64}
    # precomputed for empty plan
    MAT_n::Factorization
    MAT_x::SubArray{Float64,2,Matrix{Float64}}
end

function FirstOrderSD(model::Model)

    if !isempty(model.auxvars)
        error("Found auxiliary variables. Not yet implemented.")
    end

    ed = getevaldata(model, :firstorder)::FirstOrderMED

    vi = LittleDict{Symbol,Int}(
        var.name => ind for (ind, var) in enumerate(model.allvars)
    )

    nbck = length(ed.bck_vars)
    nfwd = length(ed.fwd_vars)
    nex = length(ed.ex_vars)
    oex = nbck + nfwd

    inds_map = Vector{Tuple{Int,Int}}(undef, nbck + nfwd + nex)
    ss_vec = Vector{Float64}(undef, nbck + nfwd + nex)
    for ((v, t), i) in ed.bck_inds
        inds_map[i] = (vi[v], t)
        ss_vec[i] = model.sstate[v][t]
    end
    for ((v, t), i) in ed.fwd_inds
        inds_map[i] = (vi[v], t)
        ss_vec[i] = model.sstate[v][t]
    end
    for ((v, t), i) in ed.ex_inds
        inds_map[oex+i] = (vi[v], t)
        ss_vec[oex+i] = model.sstate[v][t]
    end

    # compute the Schur decomposition
    qz = run_qz(ed.FWD, ed.BCK, nbck)
    # compute the system matrices
    Zbb, ZfbByZbb, R, MAT = first_order_system(qz, ed.EX, nbck, nfwd, nex)

    
    return FirstOrderSD(qz, vi,
        inds_map,
        ss_vec,
        Zbb, ZfbByZbb, R, MAT,
        # pre-compute factorization for most common case (empty plan)
        lu(MAT[:, 1:(nbck+nfwd)]),
        view(MAT, :, oex .+ (1:nex)))
end


function first_order_system(qz::QZData, EX::Matrix{Float64}, nbck::Int, nfwd::Int, nex::Int)
    oex = nbck + nfwd

    Zbb = lu(qz.Z[1:nbck, 1:nbck])
    Zbf = qz.Z[1:nbck, nbck.+(1:nfwd)]
    Zfb = qz.Z[nbck.+(1:nfwd), 1:nbck]
    Zff = qz.Z[nbck.+(1:nfwd), nbck.+(1:nfwd)]

    Tbb = qz.T[1:nbck, 1:nbck]
    Tbf = qz.T[1:nbck, nbck.+(1:nfwd)]
    # Tfb = qz.T[nbck.+(1:nfwd), 1:nbck]  # this one is 0
    Tff = lu(qz.T[nbck.+(1:nfwd), nbck.+(1:nfwd)])

    Sbb = qz.S[1:nbck, 1:nbck]
    Sbf = qz.S[1:nbck, nbck.+(1:nfwd)]
    # Sfb = qz.S[nbck.+(1:nfwd), 1:nbck]  # this one is 0
    Sff = qz.S[nbck.+(1:nfwd), nbck.+(1:nfwd)]

    QEX = qz.Q'EX

    R = vcat(-Tbb, Zff'Zfb)

    TiXf = Tff \ QEX[nbck.+(1:nfwd), :]

    MAT = zeros(nbck + nfwd, nbck + nfwd + nex)

    MAT[1:nbck, 1:nbck] = Sbb / Zbb
    # MAT[1:nbck, nbck .+(1:nfwd)] .= 0  # already zero 
    MAT[1:nbck, oex.+(1:nex)] = QEX[1:nbck, :] - (Tbf - Tbb * (Zbb \ Zbf)) * TiXf

    # MAT[nbck .+ (1:nfwd), 1:nbck] .= 0  # already zero 
    MAT[nbck.+(1:nfwd), nbck.+(1:nfwd)] = Zff'
    MAT[nbck.+(1:nfwd), oex.+(1:nex)] = TiXf

    return Zbb, Zfb / Zbb, R, MAT
end

