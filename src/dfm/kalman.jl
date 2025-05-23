##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains implementation of the api of ..Kalman for 
# DFMModels

struct DFMKalmanWks{T,Tμ,TΛ,TR,TA,TQ}
    μ::Tμ
    Λ::TΛ
    R::TR
    A::TA
    Q::TQ
    Tx::Vector{T}
    Txx::Matrix{T}
    Txx_1::Matrix{T}
    Txx_2::Matrix{T}
    Ty::Vector{T}
    Tyx::Matrix{T}
end

function DFMKalmanWks(model::DFMModel, params::DFMParams; sparse=false, sparse_A=sparse, sparse_Q=sparse)
    return DFMKalmanWks(DFM(model, params); sparse, sparse_A, sparse_Q)
end
function DFMKalmanWks(M::DFM; sparse=false, sparse_A=sparse, sparse_Q=sparse)
    NO = Kalman.kf_length_y(M)
    NS = Kalman.kf_length_x(M)
    @unpack model, params = M
    T = eltype(params)
    μ = DFMModels.get_mean!(Vector{T}(undef, NO), model, params)
    Λ = DFMModels.get_loading!(Matrix{T}(undef, NO, NS), model, params)
    A = sparse_A ? spzeros(T, NS, NS) : zeros(T, NS, NS)
    DFMModels.get_transition!(A, model, params)
    R = DFMModels.get_covariance!(Diagonal(Vector{T}(undef, NO)), model, params, Val(:Observed))
    Q = sparse_Q ? spzeros(T, NS, NS) : zeros(T, NS, NS)
    DFMModels.get_covariance!(Q, model, params, Val(:State))
    return DFMKalmanWks{T,typeof(μ),typeof(Λ),typeof(R),typeof(A),typeof(Q)}(
        μ, Λ, R, A, Q,
        Vector{T}(undef, NS),
        Matrix{T}(undef, NS, NS),
        Matrix{T}(undef, NS, NS),
        Matrix{T}(undef, NS, NS),
        Vector{T}(undef, NO),
        Matrix{T}(undef, NO, NS))
end

function TimeSeriesEcon.compare_equal(x::T, y::T; kwargs...) where {T<:DFMKalmanWks}
    for prop in (:μ, :Λ, :R, :A, :Q)
        if !compare(getproperty(x, prop), getproperty(y, prop), prop; kwargs...)
            return false
        end
    end
    return true
end

_update_wks!(wks::DFMKalmanWks, M::DFM) = _update_wks!(wks, M.model, M.params)
function _update_wks!(wks::DFMKalmanWks, model::DFMModel, params::DFMParams)
    @unpack μ, Λ, R, A, Q = wks
    DFMModels.get_mean!(μ, model, params)
    DFMModels.get_loading!(Λ, model, params)
    DFMModels.get_covariance!(R, model, params, Val(:Observed))
    DFMModels.get_transition!(A, model, params)
    DFMModels.get_covariance!(Q, model, params, Val(:State))
    return wks
end

_update_params!(M::DFM, wks::DFMKalmanWks) = (_update_params!(M.params, M.model, wks); return M)
function _update_params!(params::DFMParams, model::DFMModel, wks::DFMKalmanWks)
    @unpack μ, Λ, R, A, Q = wks
    DFMModels.set_mean!(params, model, μ)
    DFMModels.set_loading!(params, model, Λ)
    DFMModels.set_transition!(params, model, A)
    DFMModels.set_covariance!(params, model, R, Val(:Observed))
    DFMModels.set_covariance!(params, model, Q, Val(:State))
    return params
end


# function Kalman.kf_linear_stationary(H, F, Q, R, M::DFM, ::SimData, wks::DFMKalmanWks=DFMKalmanWks(M))
#     copyto!(H, wks.Λ)
#     copyto!(F, wks.A)
#     copyto!(Q, wks.Q)
#     copyto!(R, wks.R)
#     return
# end

# function Kalman.kf_predict_x!(t, xₜ, Pxₜ, Pxxₜ₋₁ₜ, xₜ₋₁, Pxₜ₋₁, M::DFM, ::AbstractMatrix, wks::DFMKalmanWks=DFMKalmanWks(M))

#     @unpack A, Q, Txx = wks
#     # xₜ = A * xₜ₋₁
#     if !isnothing(xₜ)
#         BLAS.gemv!('N', 1.0, A, xₜ₋₁, 0.0, xₜ)
#     end

#     isnothing(Pxₜ) && isnothing(Pxxₜ₋₁ₜ) && return

#     # this one is needed for both Pxₜ and Pxxₜ₋₁ₜ 
#     #    Txx = A * Pxₜ₋₁
#     mul!(Txx, A, Pxₜ₋₁)
#     # BLAS.symm!('R', 'U', 1.0, Pxₜ₋₁, A, 0.0, Txx)

#     # Pxₜ = A * Pxₜ₋₁ * A' + G * Q * G'
#     if !isnothing(Pxₜ)
#         copyto!(Pxₜ, Q) # G = I 
#         mul!(Pxₜ, Txx, transpose(A), 1.0, 1.0)
#     end

#     # Pxxₜ₋₁ₜ = Pxₜ₋₁ * A'
#     if !isnothing(Pxxₜ₋₁ₜ)
#         #    Txx already contains A * Pxₜ₋₁
#         copyto!(Pxxₜ₋₁ₜ, transpose(Txx))
#     end

#     return
# end

# function Kalman.kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, M::DFM, ::AbstractMatrix, wks::DFMKalmanWks)

#     @unpack μ, Λ, R, Tyx = wks

#     if !isnothing(yₜ)
#         # yₜ = mu + Λ xₜ
#         copyto!(yₜ, μ)
#         mul!(yₜ, Λ, xₜ, 1.0, 1.0)
#     end

#     isnothing(Pyₜ) && isnothing(Pxyₜ) && return

#     # this one is needed for both Pyₜ and Pxyₜ
#     mul!(Tyx, Λ, Pxₜ)
#     # BLAS.symm!('R', 'U', 1.0, Pxₜ, wks.Λ, 0.0, wks.Tyx)

#     if !isnothing(Pyₜ)
#         # Pyₜ = Λ Pxₜ Λ' + G * R * G'
#         copyto!(Pyₜ, R) # G = I
#         mul!(Pyₜ, Tyx, transpose(Λ), 1.0, 1.0)
#     end

#     if !isnothing(Pxyₜ)
#         # Pxyₜ = Pxₜ * Λ'
#         copyto!(Pxyₜ, transpose(Tyx))
#     end

#     return
# end

# function Kalman.kf_true_y!(t, yₜ, ::DFM, data::AbstractMatrix, ::DFMKalmanWks)
#     rng = 1:length(yₜ)
#     copyto!(transpose(yₜ), 1:1, rng, data, t:t, rng)
# end

# function Kalman.kf_true_y!(t, yₜ, M::DFM, data::SimData, ::DFMKalmanWks)
#     copyto!(yₜ, view(data, t, observed(M.model)))
# end


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

function Kalman.kf_linear_model(dfm::DFM, args...)
    model = dfm.model
    params = dfm.params
    m = KFLinearModel(dfm, args...)
    DFMModels.get_mean!(m.mu, model, params)
    DFMModels.get_loading!(m.H, model, params)
    DFMModels.get_covariance!(m.Q, model, params, Val(:Observed))
    DFMModels.get_transition!(m.F, model, params)
    DFMModels.get_covariance!(m.R, model, params, Val(:State))
    return m
end


# Kalman.dk_filter!(kf::Kalman.KFilter, Y, wks::DFMKalmanWks, args...) =
#     Kalman.dk_filter!(kf, Y, wks.μ, wks.Λ, wks.A, wks.R, wks.Q, I, args...)

# Kalman.dk_smoother!(kf::Kalman.KFilter, Y, wks::DFMKalmanWks, args...) =
#     Kalman.dk_smoother!(kf, Y, wks.μ, wks.Λ, wks.A, wks.R, wks.Q, I, args...)


#############################################################################
# The following functions provide conversion from KFData to SimData

_dfm_contemp_states_inds(dfm::DFM) = _dfm_contemp_states_inds(dfm.model)
function _dfm_contemp_states_inds(model::DFMModel)
    offset_with_lags = offset = 0
    inds = Vector{Int}(undef, nstates(model))
    for blk in values(model.components)
        ns = nstates(blk)
        L = lags(blk)
        inds[offset .+ (1:ns)] = (offset_with_lags + (L-1)*ns) .+ (1:ns)
        offset += ns
        offset_with_lags += L*ns
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
        return MVTSeries(range, [observed(dfm);states(dfm)], data)
    else
        inds = _dfm_contemp_states_inds(dfm)
        data = hcat(transpose(getproperty(kfd, y)),
            transpose(getproperty(kfd, x)[inds, :]))
        return MVTSeries(range, [observed(dfm);states_with_lags(dfm)], data)
    end
end


