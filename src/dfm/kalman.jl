##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains implementation of the api of ..Kalman for 
# DFMModels

##

Kalman.kf_is_linear(M::DFM, args...) = true

function Kalman.kf_length_x(M::DFM, args...)
    ns = 0
    for blk in values(M.model.components)
        ns += nstates(blk) * lags(blk)
    end
    return ns
end

Kalman.kf_length_y(M::DFM, args...) = nobserved(M.model)

Kalman.kf_state_noise_shaping(::DFM, user_data...) = false

function Kalman.kf_linear_model(dfm::DFM{T}, args...) where {T}
    model = dfm.model
    params = dfm.params
    return KFLinearModel(T, 
        DFMModels.get_mean(model, params),
        DFMModels.get_loading(model, params),
        DFMModels.get_transition(model, params),
        one(T)*I,
        Diagonal(DFMModels.get_covariance(model, params, Val(:Observed))),
        DFMModels.get_covariance(model, params, Val(:State))
    )
    # m = KFLinearModel(T, dfm, args...)
    # DFMModels.get_mean!(m.mu, model, params)
    # DFMModels.get_loading!(m.H, model, params)
    # DFMModels.get_transition!(m.F, model, params)
    # DFMModels.get_covariance!(m.Q, model, params, Val(:Observed))
    # DFMModels.get_covariance!(m.R, model, params, Val(:State))
    # return m
end

update_dfm_lm!(LM::KFLinearModel, M::DFM) = update_dfm_lm!(LM, M.model, M.params)
function update_dfm_lm!(LM::KFLinearModel, model::DFMModel, params::DFMParams)
    DFMModels.get_mean!(LM.mu, model, params)
    DFMModels.get_loading!(LM.H, model, params)
    DFMModels.get_transition!(LM.F, model, params)
    DFMModels.get_covariance!(LM.Q, model, params, Val(:Observed))
    DFMModels.get_covariance!(LM.R, model, params, Val(:State))
    return LM
end

update_dfm_params!(M::DFM, LM::KFLinearModel) = (update_dfm_params!(M.params, M.model, LM); return M)
function update_dfm_params!(params::DFMParams, model::DFMModel, LM::KFLinearModel)
    DFMModels.set_mean!(params, model, LM.mu)
    DFMModels.set_loading!(params, model, LM.H)
    DFMModels.set_transition!(params, model, LM.F)
    DFMModels.set_covariance!(params, model, LM.Q, Val(:Observed))
    DFMModels.set_covariance!(params, model, LM.R, Val(:State))
    return params
end


#############################################################################
# The following functions provide conversion from KFData to SimData

_dfm_contemp_states_inds(dfm::DFM) = _dfm_contemp_states_inds(dfm.model)
function _dfm_contemp_states_inds(model::DFMModel)
    offset_with_lags = offset = 0
    inds = Vector{Int}(undef, nstates(model))
    for blk in values(model.components)
        ns = nstates(blk)
        L = lags(blk)
        inds[offset.+(1:ns)] = (offset_with_lags + (L - 1) * ns) .+ (1:ns)
        offset += ns
        offset_with_lags += L * ns
    end
    return inds
end

export kfd2data
function kfd2data(kfd::Kalman.AbstractKFData, which::Symbol,
    dfm::DFM, range::AbstractUnitRange{<:MIT};
    states_with_lags::Bool=true)

    wstr = lowercase(string(which))
    if startswith(wstr, "update") || startswith(wstr, "filter")
        y = :y_pred  # we don't really have an updated y
        x = :x
    elseif startswith(wstr, "pred")
        y = :y_pred
        x = :x_pred
    elseif startswith(wstr, "smooth")
        y = :y_smooth 
        x = :x_smooth
    else
        error("Unknown :$(which); try one of :updated, :predicted, :smoothed.")
    end
    if states_with_lags
        data = hcat(transpose(getproperty(kfd, y)),
            transpose(getproperty(kfd, x)))
        return MVTSeries(range, [observed(dfm); DFMModels.states_with_lags(dfm)], data)
    else
        inds = _dfm_contemp_states_inds(dfm)
        data = hcat(transpose(getproperty(kfd, y)),
            transpose(getproperty(kfd, x)[inds, :]))
        return MVTSeries(range, [observed(dfm); DFMModels.states(dfm)], data)
    end
end


