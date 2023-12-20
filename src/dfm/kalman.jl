##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains implementation of the api of ..Kalman for 
# DFMModels

struct DFMKalman <: Kalman.AbstractKFModel
    model::DFMModel
    params::DFMParams
end
DFMKalman(dfm::DFM) = DFMKalman(dfm.model, dfm.params)

import ModelBaseEcon.DFMModels._getcoef
import ModelBaseEcon.DFMModels._getloading
import ModelBaseEcon.DFMModels._enumerate_vars

function Kalman.kf_predict_x!(t, xₜ, Pxₜ, Pxxₜ₋₁ₜ, xₜ₋₁, Pxₜ₋₁, M::DFMKalman, ::SimData)
    have_x = !isnothing(xₜ)
    have_Px = !isnothing(Pxₜ)
    have_Pxx = !isnothing(Pxxₜ₋₁ₜ)
    have_x && fill!(xₜ, 0)
    have_Px && fill!(Pxₜ, 0)
    have_Pxx && fill!(Pxxₜ₋₁ₜ, 0)

    m = M.model
    p = M.params
    current_offset = 0
    for (bname, blk) in m.components
        blags = lags(blk)
        bstates = nstates(blk)
        bpar = getproperty(p, bname)
        # todo: optimize this so we don't allocate temporary matrix each time
        AllF = zeros(Float64, bstates, bstates * blags)
        for i = 1:blags
            AllF[:, (blags-i)*bstates.+(1:bstates)] = _getcoef(blk, bpar, i)
        end
        binds = current_offset .+ (1:bstates*blags)                                 # 1:end
        binds12 = current_offset .+ (1:bstates*(blags-1))                           # 1:end-1
        binds23 = current_offset .+ (bstates+1:bstates*blags)                       # 2:end
        binds33 = current_offset .+ (bstates*blags-bstates+1:bstates*blags)         # end:end
        if have_x
            xₜ[binds33] .= AllF * xₜ₋₁[binds]
            if hasproperty(bpar, :mean)
                xₜ[binds33] += bpar.mean
            end
            if blags > 1
                xₜ[binds12] .= xₜ₋₁[binds23]
            end
        end
        if have_Px
            if blags == 1
                Pxₜ[binds, binds] = AllF * Pxₜ₋₁[binds, binds] * transpose(AllF)
            else
                # We are calculating this: 
                # [  0   I   0  ] [ P₃₃  P₃₂  P₃₁] [  0   0   F₃ᵀ]
                # [  0   0   I  ] [ P₂₃  P₂₂  P₂₁] [  I   0   F₂ᵀ]
                # [  F₃  F₂  F₁ ] [ P₁₃  P₁₂  P₁₁] [  0   I   F₁ᵀ]
                #
                # AllF = [  F₃  F₂  F₁ ]
                #
                Pxₜ[binds33, binds] = AllF * Pxₜ₋₁[binds, binds]
                Pxₜ[binds12, binds33] = transpose(Pxₜ[binds33, binds23])
                Pxₜ[binds12, binds12] = Pxₜ₋₁[binds23, binds23]
                Pxₜ[binds33, binds33] = Pxₜ[binds33, binds] * transpose(AllF)
                Pxₜ[binds33, binds12] = transpose(Pxₜ[binds12, binds33])
            end
            # add the shocks covariance  
            # should be G * Q * transpose(G), but G is identity, so
            Q = get_covariance(bpar, blk)
            Pxₜ[binds33, binds33] .+= Q
        end
        if have_Pxx
            if blags == 1
                Pxxₜ₋₁ₜ[binds, binds] = Pxₜ₋₁[binds, binds] * transpose(AllF)
            else
                # We are calculating this: 
                # [ P₃₃  P₃₂  P₃₁] [  0   0   F₃ᵀ]
                # [ P₂₃  P₂₂  P₂₁] [  I   0   F₂ᵀ]
                # [ P₁₃  P₁₂  P₁₁] [  0   I   F₁ᵀ]
                #
                # AllF = [  F₃  F₂  F₁ ]
                #
                Pxxₜ₋₁ₜ[binds, binds12] = Pxₜ₋₁[binds, binds23]
                Pxxₜ₋₁ₜ[binds, binds33] = Pxₜ₋₁[binds, binds] * transpose(AllF)
            end
        end
        current_offset += blags * bstates
    end
    return
end

function Kalman.kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, M::DFMKalman, ::SimData)
    have_y = !isnothing(yₜ)
    have_Py = !isnothing(Pyₜ)
    have_Pxy = !isnothing(Pxyₜ)

    have_y && fill!(yₜ, 0)
    have_Py && fill!(Pyₜ, 0)
    have_Pxy && fill!(Pxyₜ, 0)

    obs = M.model.observed_block
    par = M.params.observed
    yinds = _enumerate_vars(endog(obs))
    current_offset = 0
    for (bname, blk) in obs.components
        Λ = _getloading(blk, par, bname)
        byvars = obs.comp2vars[bname]
        byinds = Int[yinds[v] for v in byvars]
        blags = lags(blk)
        bstates = nstates(blk)
        bxinds = current_offset .+ (bstates*blags-bstates+1:bstates*blags)
        have_y && (yₜ[byinds] .+= Λ * xₜ[bxinds])
        TMP = Pxₜ[bxinds, bxinds] * transpose(Λ)
        have_Py && (Pyₜ[byinds, byinds] .+= Λ * TMP)
        have_Pxy && (Pxyₜ[bxinds, byinds] .+= TMP)
        current_offset += blags * bstates
    end
    have_y && (yₜ .+= par.mean)
    # add shocks covariance
    if have_Py
        byinds = Int[yinds[v] for v in keys(obs.var2shk)]
        Pyₜ[byinds, byinds] .+= get_covariance(par, obs)
    end
    return
end

function Kalman.kf_true_y!(t, yₜ, M::DFMKalman, data::SimData)
    copyto!(yₜ, view(data, t, observed(M.model)))
end

function Kalman.kf_length_x(M::DFMKalman, ::SimData)
    ns = 0
    for blk in values(M.model.components)
        ns += nstates(blk) * lags(blk)
    end
    return ns
end

Kalman.kf_length_y(M::DFMKalman, ::SimData) = nobserved(M.model)
