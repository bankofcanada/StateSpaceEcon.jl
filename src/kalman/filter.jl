##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains the main filter algorithm

struct KFilter{KFD<:AbstractKFData,ET<:Real,PT<:Integer}
    nx::Int
    ny::Int
    range::UnitRange{PT}
    x::Vector{ET}
    Px::Matrix{ET}
    x_pred::Vector{ET}
    Px_pred::Matrix{ET}
    error_y::Vector{ET}
    y_pred::Vector{ET}
    Py_pred::Matrix{ET}
    Pxy_pred::Matrix{ET}
    K::Matrix{ET}
    kfd::KFD

    function KFilter(kfd::AbstractKFData{RANGE,NS,NO,ET}) where {RANGE,NS,NO,ET}
        range = isa(RANGE, UnitRange) ? RANGE : 1:RANGE
        PT = eltype(range)
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
        new{typeof(kfd),ET,PT}(NS, NO, range, x, Px, x_pred, Px_pred,
            error_y, y_pred, Py_pred, Pxy_pred, K, kfd)
    end
end

Base.eltype(::Type{KFilter{KFD}}) where {KFD<:AbstractKFData} = eltype(KFD)

function _filter_iteration(kf::KFilter, t, KFM::AbstractKFModel, user_data...)

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
    kf_predict_x!(t, x_pred, Px_pred, x, Px, KFM, user_data...)
    # store predicted state
    kfd_setvalue!(kfd, x_pred, t, Val(:x_pred))
    kfd_setvalue!(kfd, Px_pred, t, Val(:Px_pred))

    # predict observation
    kf_predict_y!(t, y_pred, Py_pred, Pxy_pred, x_pred, Px_pred, KFM, user_data...)
    # store predicted observation
    kfd_setvalue!(kfd, y_pred, t, Val(:y_pred))
    kfd_setvalue!(kfd, Py_pred, t, Val(:Py_pred))
    kfd_setvalue!(kfd, Pxy_pred, t, Val(:Pxy_pred))

    # compute the Kalman gain:  K = Pxy_pred / Py_pred
    CPy = cholesky!(Symmetric(Py_pred, :L))
    # copyto!(kf.K, kf.Pxy_pred)  # this is a no-op since they use the same memory
    rdiv!(kf.K, CPy)
    # store Kalman gain matrix
    kfd_setvalue!(kfd, kf.K, t, Val(:K))

    # observe y
    kf_true_y!(t, y, KFM, user_data...)
    kfd_setvalue!(kfd, y, t, Val(:y))
    # observation error
    # copyto!(error_y, y) # this is a no-op since error_y and y occupy the same memory
    error_y .-= y_pred
    kfd_setvalue!(kfd, error_y, t, Val(:error_y))

    # update states based on observation error
    x .= x_pred + K * error_y
    #  Px = Px_pred - K * Py_pred * transpose(K)
    # reuse K memory 
    rmul!(K, CPy.L)  # K = K * sqrt(Py)
    Px .= Px_pred - K * transpose(K)
    # store x and Px 
    kfd_setvalue!(kfd, x, t, Val(:x))
    kfd_setvalue!(kfd, Px, t, Val(:Px))

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

function _assign_loglik(kfd::AbstractKFData{R,NS,NO,T}, t, error_y, CPy) where {R,NS,NO,T}
    if @generated
        if hasfield(kfd, :loglik)
            nc = NS * log(2 * π)
            return quote
                ldiv!(CPy.L, error_y)
                half_log_det_Py = 0
                for i = 1:NO
                    half_log_det_Py += CPy.L[i, i]
                end
                loglik = -0.5 * ($nc + half_log_det_Py + dot(error_y, error_y))
                kfd_setvalue!(kfd, loglik, t, Val(:loglik))
            end
        else
            return nothing
        end
    else
        nc = NS * log(2 * π)
        ldiv!(CPy.L, error_y)
        half_log_det_Py = 0
        for i = 1:NS
            half_log_det_Py += CPy.L[i, i]
        end
        loglik = -0.5 * (nc + half_log_det_Py + dot(error_y, error_y))
        kfd_setvalue!(kfd, loglik, t, Val(:loglik))
    end
end


function filter!(kf::KFilter,
    x0::AbstractVector, Px0::AbstractMatrix,
    KFM::AbstractKFModel, user_data...
)
    kf.x .= x0
    kf.Px .= Px0

    for t = kf.range
        _filter_iteration(kf, t, KFM, user_data...)
    end

    return kf
end
function filter!(kfd::AbstractKFData, args...)
    kf = KFilter(kfd)
    filter!(kf, args...)
    return kf.kfd
end

filter(N::Integer, args...) = filter(1:N, args...)
filter(range::UnitRange, x0::AbstractVector, Px0::AbstractMatrix, KFM::AbstractKFModel, user_data...) =
    filter!(KFDataFilter(range, KFM, user_data...), x0, Px0, KFM, user_data...)

