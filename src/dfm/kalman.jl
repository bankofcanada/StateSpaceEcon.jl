##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


Kalman.nobserved(model::DFMModel) = ModelBaseEcon.nobserved(model)
Kalman.nobservedshocks(model::DFMModel) = ModelBaseEcon.nobservedshocks(model)
Kalman.nstates(model::DFMModel) = ModelBaseEcon.nstates(model)
Kalman.nstateshocks(model::DFMModel) = ModelBaseEcon.nstateshocks(model)


function Kalman.eval_transition(model::DFMModel,
    state::AbstractVector{Float64},
    stateshock::AbstractVector{Float64},
    period::Integer)

    NS = Kalman.nstates(model)
    NSS = Kalman.nstateshocks(model)
    @assert length(state) == NS "Dimensions don't match."
    @assert length(stateshock) == NSS "Dimensions don't match."
    newstate = zeros(NS)
    F = zeros(NS, NS)  # Jacobian of transition equation w.r.t. states
    G = zeros(NS, NSS) # Jacobian of transition equation w.r.t. state shocks
    last_ind_s = last_ind_ss = 0
    # Assemble global matrices from block matrices and compute new states on the fly
    # Note that blocks are mutually decoupled, so there's no accumulation here -- just
    #   assign each block matrix within the global matrix at the appropriate location
    for fb in ModelBaseEcon._blocks(model)
        block_F, block_G = ModelBaseEcon.block_transition(fb)
        ns = size(block_F, 1)   # size(blk_F) == (fb.nstates, fb.nstates)
        nss = size(block_G, 2)  # size(blk_G) == (fb.nstates, fb.nstateshocks)
        ind_s = last_ind_s .+ (1:ns)    # index range within model_F and rows of model_G
        ind_ss = last_ind_ss .+ (1:nss) # index range for columns of model_G
        newstate[ind_s] = block_F * state[ind_s] + block_G * stateshock[ind_ss]
        F[ind_s, ind_s] = block_F
        G[ind_s, ind_ss] = block_G
        last_ind_s += ns
        last_ind_ss += nss
    end
    return newstate, F, G
end


function Kalman.stateshocks_covariance(model::DFMModel, period::Integer)
    NSS = Kalman.nstateshocks(model)
    Q = zeros(NSS, NSS)
    last_ind_ss = 0
    for fb in ModelBaseEcon._blocks(model)
        block_Q = ModelBaseEcon.block_covariance(fb)
        local nss = size(block_Q, 1)
        ind_ss = last_ind_ss .+ (1:nss)
        Q[ind_ss, ind_ss] = block_Q
        last_ind_ss += nss
    end
    return Symmetric(Q)
end

function Kalman.eval_observation(model::DFMModel,
    state::AbstractVector{Float64},
    observedshock::AbstractVector{Float64},
    period::Integer)

    NO = Kalman.nobserved(model)
    NOS = Kalman.nobservedshocks(model)
    NS = Kalman.nstates(model)
    @assert length(state) == NS "Dimensions don't match."
    @assert length(observedshock) == NOS "Dimensions don't match."
    observed = copy(model.mean)
    @assert length(observed) == NO "Dimensions don't match."
    H = zeros(NO, NS)
    G = zeros(NO, NOS)
    # Assemble the global H from the block matrices. The states (columns) do not overlap
    #   but the observed (columns) might.  So when computing the new observed, 
    #   we must accumulate, i.e., use += rather than =
    last_ind_s = 0
    mvars = ModelBaseEcon.observed(model)
    for fb in ModelBaseEcon._blocks(model)
        block_H = ModelBaseEcon.block_observation(fb)
        local ns = size(block_H, 2)
        ind_s = last_ind_s .+ (1:ns)
        ind_o = indexin(ModelBaseEcon.observed(fb), mvars)
        H[ind_o, ind_s] = block_H
        observed[ind_o] += block_H * state[ind_s]
        last_ind_s += ns
    end
    # add observation shock
    if NOS > 0
        ind_o = indexin(ModelBaseEcon.observed_no_ic(model), mvars)
        G[ind_o, :] = I(NOS)
        observed[ind_o] += model.covariance[] * observedshock
    end
    return observed, H, G

end


function Kalman.observedshocks_covariance(model::DFMModel, period::Integer)
    NOS = Kalman.nobservedshocks(model)
    R = zeros(NOS, NOS)
    if NOS > 0
        R[:,:] = model.covariance[]
    end
    return R
end


