##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################



"""
    _assign_res2(kfd::AbstractKFData, t, error_y)

Internal function that calculates the squared residual from a given vector of
observed errors and assigns the value in the given data container
"""
function _assign_res2(kfd::AbstractKFData, t, error_y)
    if @generated
        if hasfield(kfd, :res2)
            return :(kfd_setvalue!(kfd, dot(error_y, error_y), t, Val(:res2)))
        else
            return nothing
        end
    else
        if hasfield(typeof(kfd), :res2)
            kfd[t, :res2] = dot(error_y, error_y)
        end
    end
end

function _calc_loglik(error_y::AbstractVector, CPy::Cholesky)
    ny = length(error_y)
    ny == 0 && return 0.0
    nc = ny * log(2 * π)
    ldiv!(CPy.L, error_y)
    half_log_det_Py = 0
    for i = 1:ny
        half_log_det_Py += log(CPy.factors[i, i])
    end
    loglik = -0.5 * (nc + 2half_log_det_Py + dot(error_y, error_y))
    return loglik
end

"""
    _assign_loglik(kfd::AbstractKFData, t, error_y, CPy::Cholesky)

Internal function that calculates the log likelihood from given vector of
observation errors and observation covariance matrix and assigns the value in
the given Kalman data container.
"""
function _assign_loglik(kfd::AbstractKFData, t, error_y, CPy::Cholesky)
    if @generated
        if hasfield(kfd, :loglik)
            return :(kfd_setvalue!(kfd, _calc_loglik(error_y, CPy), t, Val(:loglik)))
        else
            return nothing
        end
    else
        if hasfield(typeof(kfd), :loglik)
            kfd[t, :loglik] = _calc_loglik(error_y, CPy)
        end
    end
end


"""
    _symm!(A)

Force matrix `A` to be symmetric by overwriting it with `0.5 (A + Aᵀ)`.
This is useful to stabilize the algorithm when a matrix is known to be
symmetric, but may lose this property due accumulation of round off and
truncation errors.
"""
function _symm!(A::AbstractMatrix)
    m, n = size(A)
    @assert m == n && axes(A) == (Base.OneTo(m), Base.OneTo(n))
    @inbounds for i = 1:m
        for j = i+1:m
            v = 0.5 * (A[i, j] + A[j, i])
            A[i, j] = v
            A[j, i] = v
        end
    end
    return A
end
_symm!(::SparseMatrixCSC) = error("Not implemented for sparse matrices")
