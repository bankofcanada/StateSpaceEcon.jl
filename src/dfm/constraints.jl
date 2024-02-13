##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains data structures and methods for handling constraints on 
# the loadings matrix and the factors' AR matrix of the factors during the EM algorithm


struct DFMConstraint{T,TW<:AbstractMatrix{T},Tq<:AbstractVector{T}}
    # constraint on matrix A is in the form
    #     W * vec(A) = q
    # constraints data  
    numcols::Int    #   == number of columns of A, the matrix being constrained
    W::TW           #   == constraint matrix
    q::Tq           #   == constraint right-hand-side
    # pre-allocated work arrays used in the algorithms
    Tc::Vector{T}
    BT::Matrix{T}
    Tcr::Matrix{T}
    Tcc::Matrix{T}
    # inner constructor allocates work arrays in the correct dimensions
    function DFMConstraint(numcols::Integer, W::AbstractMatrix{T}, q::AbstractVector{T}) where {T}
        ncons, nels = size(W)
        if length(q) != ncons
            throw(DimensionMismatch("Expected equal number of columns in W and q. Got size(W) = $(size(W)) and size(q) = $(size(q))"))
        end
        numrows, rem = divrem(nels, numcols)
        if rem != 0
            throw(ArgumentError("size(W, 2) = $(nels) is incompatible with numcols = $(numcols)"))
        end
        Tc = Vector{T}(undef, ncons)
        BT = Matrix{T}(undef, ncons, nels)
        Tcr = Matrix{T}(undef, ncons, numrows)
        Tcc = Matrix{T}(undef, ncons, ncons)
        return new{T,typeof(W),typeof(q)}(numcols, W, q, Tc, BT, Tcr, Tcc)
    end
end

function DFMConstraint(A::AbstractMatrix{T};
    add_ncons::Integer=0, to_estimate::Function=isnan) where {T}

    if add_ncons < 0
        throw(ArgumentError("add_ncons must not be negative."))
    end

    # entries of A are NaN  where they need to be estimated and not NaN where 
    # they are known.
    #
    # We build W and q such that constraints are of the form:
    #     W * vec(A) = q 
    #
    # number of columns equals the total number of entries 
    nels = length(A)
    # number of rows equals the number of fixed entries (ones that will not be estimated)
    ncons = nels - sum(to_estimate, A)
    # allocate matrix W
    W = Matrix{T}(undef, ncons + add_ncons, nels)
    # allocate vector q
    q = Vector{T}(undef, ncons + add_ncons)
    # let's do it
    fill!(W, zero(T))
    fill!(q, zero(T))
    r = c = 0
    while r < ncons
        c = c + 1
        while to_estimate(A[c])
            c = c + 1
        end
        r = r + 1
        W[add_ncons+r, c] = one(T)
        q[add_ncons+r] = A[c]
    end
    return DFMConstraint(size(A, 2), W, q)
end

function DFMConstraint(M::DFM, ::Val{:Λ}; add_ncons::Integer=-1, kwargs...)
    NO = Kalman.kf_length_y(M)
    NS = Kalman.kf_length_x(M)
    Λ = Matrix{Float64}(undef, NO, NS)
    DFMModels.get_loading!(Λ, M.model, M.params)
    add_ncons < 0 && (add_ncons = DFMModels.get_loading_ncons(M.model, M.params))
    ret = DFMConstraint(Λ; add_ncons, kwargs...)
    add_ncons > 0 && DFMModels.get_loading_cons!(ret.W, ret.q, M.model, M.params)
    return ret
end

function DFMConstraint(M::DFM, ::Val{:A}; kwargs...)
    NS = Kalman.kf_length_x(M)
    A = Matrix{Float64}(undef, NS, NS)
    DFMModels.get_transition!(A, M.model, M.params)
    return DFMConstraint(A; kwargs...)
end


_apply_constraint!(M::AbstractMatrix, ::Nothing, args...) = M
function _apply_constraint!(M::AbstractMatrix, cons::DFMConstraint{T}, 
        cXᵀX::Cholesky, Σ::AbstractMatrix) where {T}
    @unpack numcols, W, q, Tc, BT, Tcr, Tcc = cons
    ncons, nels = size(W)
    ncons > 0 || return M

    if nels != length(M) || numcols != size(M, 2)
        throw(ArgumentError("incompatible dimensions of matrix and constraints"))
    end

    # constraint residuals
    Tc[:] = q
    mul!(Tc, W, vec(M), -1.0, 1.0)
    maximum(abs, Tc) < 100eps() && return M

    # @info "Applying Λ-constraints"

    # The formula is
    #    resid = W * vec(M) - q
    #    A = kron(inv(XᵀX), Σ)
    #    B = A * transpose(W)
    #    C = W * B
    #    ϰ = inv(C) * resid
    #    M = M + B * ϰ

    # construct the inverse of XᵀX from its Cholesky factor
    # done in-place, overwriting the Cholesky factor
    iXᵀX = cXᵀX.factors
    LAPACK.potri!(cXᵀX.uplo, iXᵀX)
    iXᵀX = Symmetric(iXᵀX, Symbol(cXᵀX.uplo))

    # BT will actually contain Bᵀ = W * Aᵀ (but A is symmetric, so there)
    # we compute BT = W * A from the Kronecker factors of A, without explicitly constructing A
    fill!(BT, zero(T))
    BT3 = reshape(BT, ncons, :, numcols)  # BT3 provides a separate view into the part of BT for each column of M
    W3 = reshape(W, ncons, :, numcols)   # W3 does the same for W
    for i = 1:numcols
        mul!(Tcr, view(W3, :, :, i), Σ)
        for j = 1:numcols
            BLAS.axpy!(iXᵀX[i, j], Tcr, view(BT3, :, :, j))
        end
    end

    # C = W * B = W * (Bᵀ)ᵀ  -  Tcc contains C
    mul!(Tcc, W, transpose(BT))
    # ϰ = C \ resid  note: since C is SPD, the fastest inverse is via Cholesky
    #   Tc contains resid before and ϰ after the next line
    # cTcc = cholesky!(Symmetric(Tcc, :U))
    # Hmmm, actually cTcc may be S-semi-PD. QR with pivoting seems to work
    cTcc = qr!(Tcc, ColumnNorm())
    # cTcc = cholesky!(Symmetric(Tcc, :U), check = false)
    # if cTcc.info != 0
    #     mul!(Tcc, W, transpose(BT))
    #     Tcc += √eps(one(T))*I(ncons)
    #     cTcc = cholesky!(Symmetric(Tcc, :U))
    # end
    ldiv!(cTcc, Tc)
    # M = M + B * ϰ
    mul!(vec(M), transpose(BT), Tc, 1.0, 1.0)

    return M
end


