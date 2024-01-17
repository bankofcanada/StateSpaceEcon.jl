##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
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

function DFMKalmanWks(model::DFMModel, params::DFMParams)
    return DFMKalmanWks(DFM(model, params))
end
function DFMKalmanWks(M::DFM)
    NO = Kalman.kf_length_y(M)
    NS = Kalman.kf_length_x(M)
    @unpack model, params = M
    T = eltype(params)
    μ = copyto!(Vector{T}(undef, NO), params.observed.mean)
    Λ = DFMModels.fill_loading!(Matrix{T}(undef, NO, NS), model, params)
    A = DFMModels.fill_transition!(Matrix{T}(undef, NS, NS), model, params)
    R = DFMModels.fill_covariance!(Matrix{T}(undef, NO, NO), model, params, Val(:Observed))
    Q = DFMModels.fill_covariance!(Matrix{T}(undef, NS, NS), model, params, Val(:State))
    return DFMKalmanWks(μ, Λ, R, A, Q, similar(A), similar(Λ))
end

DFMKalmanWks_update!(wks::DFMKalmanWks, M::DFM) = DFMKalmanWks_update!(wks, M.model, M.params)
function DFMKalmanWks_update!(wks::DFMKalmanWks, model::DFMModel, params::DFMParams)
    @unpack μ, Λ, R, A, Q = wks
    copyto!(μ, params.observed.mean)
    DFMModels.fill_loading!(Λ, model, params)
    DFMModels.fill_covariance!(R, model, params, Val(:Observed))
    DFMModels.fill_transition!(A, model, params)
    DFMModels.fill_covariance!(Q, model, params, Val(:State))
    return wks
end

struct DFMConstraints
    WΛ
    qΛ
    WA
    qA
end

function _make_W_q(Mat, to_estimate=isnan)
    # eltype(Mat) is numeric
    # entries of Mat are NaN  where they need to be estimated and not NaN where 
    # they are known.
    #
    # We build W and q such that constraints are of the form:
    #     W * vec(Mat) = q 
    #
    # number of columns equals the total number of entries 
    nc = length(Mat)
    # number of rows equals the number of fixed entries (ones that will not be estimated)
    nr = nc - sum(to_estimate, Mat)
    T = eltype(Mat)
    # allocate matrix W
    W = Matrix{T}(undef, nr, nc)
    # allocate vector q
    q = Vector{T}(undef, nr)
    # let's do it
    fill!(W, zero(T))
    r = c = 0
    while r < nr
        c = c + 1
        while to_estimate(Mat[c])
            c = c + 1
        end
        r = r + 1
        W[r, c] = one(T)
        q[r] = Mat[c]
    end
    return W, q
end

function DFMConstraints(wks::DFMKalmanWks, to_estimate=isnan)
    @unpack Λ, A = wks
    WΛ, qΛ = _make_W_q(Λ, to_estimate)
    WA, qA = _make_W_q(A, to_estimate)
    return DFMConstraints(WΛ, qΛ, WA, qA)
end

function DFMConstraints(M::DFM, wks::DFMKalmanWks, to_estimate=isnan)
    @unpack model, params = M
    DFMKalmanWks_update!(wks, model, fill!(similar(params, Float64), NaN))
    cons = DFMConstraints(wks, to_estimate)
    DFMKalmanWks_update!(wks, model, params)
    return cons
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

