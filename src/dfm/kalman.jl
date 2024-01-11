##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains implementation of the api of ..Kalman for 
# DFMModels

struct DFMKalmanWks
    μ
    Λ
    R
    A
    Q
    Txx
    Tyx
end

function DFMKalmanWks(M::DFM)
    NO = Kalman.kf_length_y(M)
    NS = Kalman.kf_length_x(M)
    μ = copyto!(Vector{Float64}(undef, NO), M.params.observed.mean)
    Λ = DFMModels.fill_loading!(Matrix{Float64}(undef, NO, NS), M.model, M.params)
    A = DFMModels.fill_transition!(Matrix{Float64}(undef, NS, NS), M.model, M.params)
    R = DFMModels.fill_covariance!(Matrix{Float64}(undef, NO, NO), M.model, M.params, Val(:Observed))
    Q = DFMModels.fill_covariance!(Matrix{Float64}(undef, NS, NS), M.model, M.params, Val(:State))
    return DFMKalmanWks(μ, Λ, R, A, Q, similar(A), similar(Λ))
end

Kalman.kf_islinear(M::DFM, ::SimData, wks::DFMKalmanWks=DFMKalmanWks(M)) = true
Kalman.kf_isstationary(M::DFM, ::SimData, wks::DFMKalmanWks=DFMKalmanWks(M)) = true

function Kalman.kf_linear_stationary(H, F, Q, R, M::DFM, ::SimData, wks::DFMKalmanWks=DFMKalmanWks(M))
    copyto!(H, wks.Λ)
    copyto!(F, wks.A)
    copyto!(Q, wks.Q)
    copyto!(R, wks.R)
    return
end

function Kalman.kf_predict_x!(t, xₜ, Pxₜ, Pxxₜ₋₁ₜ, xₜ₋₁, Pxₜ₋₁, M::DFM, ::AbstractMatrix, wks::DFMKalmanWks=DFMKalmanWks(M))

    # xₜ = A * xₜ₋₁
    if !isnothing(xₜ)
        BLAS.gemv!('N', 1.0, wks.A, xₜ₋₁, 0.0, xₜ)
    end

    isnothing(Pxₜ) && isnothing(Pxxₜ₋₁ₜ) && return

    # this one is needed for both Pxₜ and Pxxₜ₋₁ₜ 
    #    Txx = A * Pxxₜ₋₁ₜ
    BLAS.symm!('R', 'U', 1.0, Pxₜ₋₁, wks.A, 0.0, wks.Txx)

    # Pxₜ = A * Pxₜ₋₁ * A' + G * Q * G'
    if !isnothing(Pxₜ)
        #   Pxₜ = Txx * A'
        BLAS.gemm!('N', 'T', 1.0, wks.Txx, wks.A, 0.0, Pxₜ)
        #   Pxₜ = Pxₜ + G * Q * G'
        Pxₜ .+= wks.Q # G = I
    end

    # Pxxₜ₋₁ₜ = Pxₜ₋₁ * A'
    if !isnothing(Pxxₜ₋₁ₜ)
        #    Txx already contains A * Pxₜ₋₁
        Pxxₜ₋₁ₜ .= transpose(wks.Txx)
    end

    return
end

function Kalman.kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, M::DFM, ::AbstractMatrix, wks::DFMKalmanWks)

    if !isnothing(yₜ)
        # yₜ = mu + Λ xₜ
        BLAS.gemv!('N', 1.0, wks.Λ, xₜ, 0.0, yₜ)
        yₜ .+= M.params.observed.mean
    end

    isnothing(Pyₜ) && isnothing(Pxyₜ) && return

    # this one is needed for both Pyₜ and Pxyₜ
    BLAS.symm!('R', 'U', 1.0, Pxₜ, wks.Λ, 0.0, wks.Tyx)

    if !isnothing(Pyₜ)
        # Pyₜ = Λ Pxₜ Λ' + G * R * G'
        BLAS.gemm!('N', 'T', 1.0, wks.Tyx, wks.Λ, 0.0, Pyₜ)
        Pyₜ .+= wks.R # G = I
    end

    if !isnothing(Pxyₜ)
        # Pxyₜ = Pxₜ * Λ'
        Pxyₜ .= transpose(wks.Tyx)
    end
    return
end

function Kalman.kf_true_y!(t, yₜ, M::DFM, data::AbstractMatrix, ::DFMKalmanWks)
    rng = 1:length(yₜ)
    copyto!(transpose(yₜ), 1:1, rng, data, t:t, rng)
end

function Kalman.kf_true_y!(t, yₜ, M::DFM, data::SimData, ::DFMKalmanWks)
    copyto!(yₜ, view(data, t, observed(M.model)))
end

function Kalman.kf_length_x(M::DFM, args...)
    ns = 0
    for blk in values(M.model.components)
        ns += nstates(blk) * lags(blk)
    end
    return ns
end

Kalman.kf_length_y(M::DFM, args...) = nobserved(M.model)

