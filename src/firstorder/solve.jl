##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

import ModelBaseEcon.FirstOrderMED
import ModelBaseEcon.LittleDict

function solve!(m::Model)
    firstorder!(m)
    m.solverdata = FirstOrderSD(m)
    return m
end

struct FirstOrderSD
    F::Factorization
    # translations from FO variable index to Var-Lag pair
    inds_map::Vector{Tuple{Int,Int}}
    # steady state (or point of linearization)
    ss_vec::Vector{Float64}
    # various matrices 
    Zbb::Factorization
    R::Matrix{Float64}
    MAT::Matrix{Float64}
    # precomputed for empty plan
    MAT_n::Factorization
    MAT_x::SubArray{Float64,2,Matrix{Float64}}
end

# function _make_inds_map(inds, vi)
#     inds_map = Vector{Tuple{Int,Int}}(undef, length(inds))
#     for ((v, t), i) in inds
#         inds_map[i] = (vi[v], t)
#     end
#     return inds_map
# end
function FirstOrderSD(model::Model)

    if !isempty(model.auxvars)
        error("Found auxiliary variables. Not yet implemented.")
    end

    ed = model.evaldata::FirstOrderMED

    # compute the Schur decomposition
    # TODO: use low-level call to LAPACK to compute and re-order in a single call
    F = schur(ed.FWD, ed.BCK)
    ordschur!(F, @. abs(F.α) > abs(F.β))
    vi = LittleDict{Symbol,Int}(
        var.name => ind for (ind, var) in enumerate(model.allvars)
    )

    nbck = length(ed.bck_vars)
    nfwd = length(ed.fwd_vars)
    nex = length(ed.ex_vars)
    ex_offset = nbck + nfwd

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
        inds_map[ex_offset+i] = (vi[v], t)
        ss_vec[ex_offset+i] = model.sstate[v][t]
    end

    S, T, Q, Z, α, β = F

    n_stable = sum(zip(α, β)) do ((a, b))
        abs(a) * (1.0 + √eps(1.0)) > abs(b)
    end
    n_unstable = sum(zip(α, β)) do ((a, b))
        abs(b) * (1.0 + √eps(1.0)) > abs(a)
    end
    if n_stable < nbck
        error("We have only $n_stable stable eigen-values while we need at least $nbck.")
    end
    if n_unstable < nfwd
        error("We have only $n_unstable unstable eigen-values while we need at least $nfwd.")
    end

    Zbb = lu(Z[1:nbck, 1:nbck])
    Zbf = Z[1:nbck, nbck.+(1:nfwd)]
    Zfb = Z[nbck.+(1:nfwd), 1:nbck]
    Zff = Z[nbck.+(1:nfwd), nbck.+(1:nfwd)]

    Tbb = T[1:nbck, 1:nbck]
    Tbf = T[1:nbck, nbck.+(1:nfwd)]
    # Tfb = T[nbck.+(1:nfwd), 1:nbck]  # this one is 0
    Tff = lu(T[nbck.+(1:nfwd), nbck.+(1:nfwd)])

    Sbb = S[1:nbck, 1:nbck]
    Sbf = S[1:nbck, nbck.+(1:nfwd)]
    # Sfb = S[nbck.+(1:nfwd), 1:nbck]  # this one is 0
    Sff = S[nbck.+(1:nfwd), nbck.+(1:nfwd)]

    QEX = Q'ed.EX

    R = vcat(-Tbb, Zff'Zfb)

    TiXf = Tff \ QEX[nbck.+(1:nfwd), :]

    MAT = zeros(nbck + nfwd, nbck + nfwd + nex)

    MAT[1:nbck, 1:nbck] = Sbb / Zbb
    # MAT[1:nbck, nbck .+(1:nfwd)] .= 0
    MAT[1:nbck, ex_offset.+(1:nex)] = QEX[1:nbck, :] - (Tbf - Tbb * (Zbb \ Zbf)) * TiXf

    # MAT[nbck .+ (1:nfwd), 1:nbck] .= 0
    MAT[nbck.+(1:nfwd), nbck.+(1:nfwd)] = Zff'
    MAT[nbck.+(1:nfwd), ex_offset.+(1:nex)] = TiXf



    return FirstOrderSD(F, inds_map, ss_vec,
        Zbb, R, MAT,
        lu(MAT[:, 1:(nbck+nfwd)]),
        view(MAT, :, ex_offset .+ (1:nex)))
end



