##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains the main filter algorithm

struct KFilter{KFD<:AbstractKFData,ET<:Real}
    nx::Int
    ny::Int
    range::UnitRange
    x::Vector{ET}
    Px::Matrix{ET}
    x_pred::Vector{ET}
    Px_pred::Matrix{ET}
    error_y::Vector{ET}
    y_pred::Vector{ET}
    Py_pred::Matrix{ET}
    Pxy_pred::Matrix{ET}
    K::Matrix{ET}
    x_smooth::Vector{ET}
    Px_smooth::Matrix{ET}
    Pxx_smooth::Matrix{ET}
    y_smooth::Vector{ET}
    Py_smooth::Matrix{ET}
    J::Matrix{ET}
    kfd::KFD

    function KFilter(kfd::AbstractKFData{RANGE,NS,NO,ET}) where {RANGE,NS,NO,ET}
        range = isa(RANGE, UnitRange) ? RANGE : 1:RANGE
        x = zeros(ET, NS)
        Px = zeros(ET, NS, NS)
        x_pred = zeros(ET, NS)
        Px_pred = zeros(ET, NS, NS)
        error_y = zeros(ET, NO)
        y_pred = zeros(ET, NO)
        Py_pred = zeros(ET, NO, NO)
        # Pxy and K occupy the same memory.
        # We can get away with it because Pxy is only used to compute K
        K = Pxy_pred = zeros(ET, NS, NO)
        # smoother things
        x_smooth = zeros(ET, NS)
        Px_smooth = zeros(ET, NS, NS)
        y_smooth = zeros(ET, NO)
        Py_smooth = zeros(ET, NO, NO)
        # Pxx and J occupy the same memory.
        # We can get away with it because Pxx is only used to compute J
        J = Pxx_smooth = zeros(ET, NS, NS)
        new{typeof(kfd),ET}(NS, NO, range,
            x, Px, x_pred, Px_pred,
            error_y, y_pred, Py_pred, Pxy_pred, K,
            x_smooth, Px_smooth, Pxx_smooth, y_smooth, Py_smooth, J, kfd)
    end
end

Base.eltype(::Type{KFilter{KFD}}) where {KFD<:AbstractKFData} = eltype(KFD)

function _filter_iteration(kf::KFilter, t, model, user_data...)

    kfd = kf.kfd

    x = kf.x
    Px = kf.Px

    x_pred = kf.x_pred
    Px_pred = kf.Px_pred

    y = error_y = kf.error_y  # y and error_y can use the same memory
    y_pred = kf.y_pred
    Py_pred = kf.Py_pred
    Pxy_pred = K = kf.K  # K and Pxy_pred occupy the same memory

    # predict state
    kf_predict_x!(t, x_pred, Px_pred, nothing, x, Px, model, user_data...)
    # store predicted state
    @kfd_set! kfd t x_pred Px_pred

    # predict observation
    kf_predict_y!(t, y_pred, Py_pred, Pxy_pred, x_pred, Px_pred, model, user_data...)
    # store predicted observation
    @kfd_set! kfd t y_pred Py_pred Pxy_pred

    # compute the Kalman gain:  K = Pxy_pred / Py_pred
    CPy = cholesky!(Symmetric(Py_pred, :L))
    # copyto!(K, Pxy_pred)  # this is a no-op since they use the same memory
    rdiv!(K, CPy)
    # store Kalman gain matrix
    @kfd_set! kfd t K

    # observe y
    kf_true_y!(t, y, model, user_data...)
    # observation error
    # copyto!(error_y, y) # this is a no-op since error_y and y occupy the same memory
    error_y .-= y_pred
    @kfd_set! kfd t y error_y

    # update states based on observation error
    x .= x_pred + K * error_y
    #  Px = Px_pred - K * Py_pred * transpose(K)
    # reuse K memory 
    rmul!(K, CPy.L)  # K = K * sqrt(Py)
    Px .= Px_pred - K * transpose(K)
    # store x and Px 
    @kfd_set! kfd t x Px

    # compute likelihood and residual-squared
    _assign_res2(kfd, t, error_y)
    _assign_loglik(kfd, t, error_y, CPy)

    return x_pred, Px_pred, y_pred, Py_pred, x, Px
end

function _assign_res2(kfd::AbstractKFData, t, error_y)
    if @generated
        if hasfield(kfd, :res2)
            return :(kfd_setvalue!(kfd, dot(error_y, error_y), t, Val(:res2)))
        else
            return nothing
        end
    else
        return kfd_setvalue!(kfd, dot(error_y, error_y), t, Val(:res2))
    end
end

function _assign_loglik(kfd::AbstractKFData{R,NS,NO,T}, t, error_y, CPy::Cholesky) where {R,NS,NO,T}
    if @generated
        if hasfield(kfd, :loglik)
            nc = NO * log(2 * π)
            return quote
                if CPy.uplo == 'L'
                    ldiv!(CPy.L, error_y)
                else
                    rdiv!(error_y, CPy.U)
                end
                half_log_det_Py = 0
                for i = 1:NO
                    half_log_det_Py += log(CPy.factors[i, i])
                end
                loglik = -0.5 * ($nc + 2half_log_det_Py + dot(error_y, error_y))
                @kfd_set! kfd t loglik
            end
        else
            return nothing
        end
    else
        nc = NO * log(2 * π)
        if CPy.uplo == 'L'
            ldiv!(CPy.L, error_y)
        else
            rdiv!(error_y, CPy.U)
        end
        half_log_det_Py = 0
        for i = 1:NO
            half_log_det_Py += log(CPy.factors[i, i])
        end
        loglik = -0.5 * (nc + 2half_log_det_Py + dot(error_y, error_y))
        @kfd_set! kfd t loglik
    end
end


function filter!(kf::KFilter,
    x0::AbstractVector, Px0::AbstractMatrix,
    model, user_data...
)
    kf.x .= x0
    kf.Px .= Px0

    for t = kf.range
        _filter_iteration(kf, t, model, user_data...)
    end

    return kf
end
function filter!(kfd::AbstractKFData, args...)
    kf = KFilter(kfd)
    filter!(kf, args...)
    return kf.kfd
end

filter(N::Integer, args...) = filter(1:N, args...)
filter(range::UnitRange, x0::AbstractVector, Px0::AbstractMatrix, model, user_data...) =
    filter!(KFDataFilter(range, model, user_data...), x0, Px0, model, user_data...)

