##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# this file contains data structures and methods for handling constraints on 
# the loadings matrix and the factors' AR matrices during the EM algorithm

"""
    struct EM_MatrixConstraint{T<:Real, p, r, n, m, p} ... end

A structure representing a linear constraint on the entries of a matrix applied
during estimation based on the EM algorithm.

Constraint is of the form
```latex
    W * vec(A) = q
```
The matrix being constrained A has size n×m. Then vec(A) is a column vector of
length nm, in which the columns of A are stacked vertically.

The constraint matrix W is of size p×r, where r=nm and p is the number of 
constraint equations. 

The constraint vector q has length p.

This 


"""
struct EM_MatrixConstraint{T<:Real, NCONS, NELS, NROWS, NCOLS,
    TW<:AbstractMatrix{T},Tq<:AbstractVector{T}}
    # Vector q has length NCONS
    # constraint on matrix A is in the form
    #     W * vec(A) = q

    # NCONS, NELS - size of W
    # NROWS, NCOLS - size of A.
    # Matrix W has dimension NCONS-by-NELS, where NELS = NROWS*NCOLS

    # constraints data  
    W::TW           #   == constraint matrix
    q::Tq           #   == constraint right-hand-side
    # pre-allocated work arrays used in the algorithms
    Tc::MVector{NCONS, T}
    Tcr::MMatrix{NCONS, NROWS, T}
    Tcc::MMatrix{NCONS, NCONS, T}
    BT::MMatrix{NCONS, NELS, T}

    # inner constructor allocates work arrays in the correct dimensions
    function EM_MatrixConstraint(ncols::Integer, W::AbstractMatrix{T}, q::AbstractVector{T}) where {T}
        # NOTE: the matrix being constrained has dimensions nrows × ncols, for a total of nels=nrows*ncols elements
        #   Matrix W has dimension ncons × nels - one row for each constraint and a column for each element of the constrained matrix
        #   We deduce nrows from ncons (given directly) and nels (from second dimension of W)
        ncols == 0 && return nothing
        ncons, nels = size(W)
        nrows, rem = divrem(nels, ncols)
        @assert rem == 0 "Inconsistent number of columns in W. Expected multiple of $ncols, got $nels"
        ncons == 0 && return nothing
        return new{T, ncons, nels, nrows, ncols, typeof(W), typeof(q)}(W, q, 
            MVector{ncons, T}(undef), 
            MMatrix{ncons, nrows, T}(undef), 
            MMatrix{ncons, ncons, T}(undef),
            MMatrix{ncons, nels, T}(undef))
    end
end

const Maybe_EM_MatrixConstraint{T} = Union{Nothing,EM_MatrixConstraint{T}}

"""
    em_apply_constraint!(M, ::Nothing, ...) = M
    em_apply_constraint!(M, cons::EM_MatrixConstraint, ...)

Modify matrix M in place to enforce the given constraint. This is done in the
course of the EM algorithm. The minimization problems is first solved
unconstrained, then that solution is updated to satisfy the constraint is such a
way that the result is the solution of the constrained minimization. This
function performs this update.

This is an internal function.
"""
function em_apply_constraint! end

em_apply_constraint!(M::AbstractMatrix{T}, ::Nothing, args...) where {T<:Real} = M
function em_apply_constraint!(M::AbstractMatrix{T}, cons::EM_MatrixConstraint{T, NCONS, NELS, NROWS, NCOLS},
    cXᵀX::Cholesky{T}, Σ::AbstractMatrix{T}) where {T, NCONS, NELS, NROWS, NCOLS}

    NCONS > 0 || return M
    
    if (NROWS, NCOLS) != size(M) 
        throw(ArgumentError("Incompatible dimensions of matrix and constraints"))
    end

    @unpack W, q, Tc, Tcr, Tcc, BT = cons

    # constraint residuals
    Tc[:] = q
    mul!(Tc, W, vec(M), -1.0, 1.0)
    # if residuals are sufficiently small, constraint is not violated, so we're done here
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
    BT3 = reshape(BT, NCONS, :, NCOLS)  # BT3 provides a separate view into the part of BT for each column of M
    W3 = reshape(W, NCONS, :, NCOLS)   # W3 does the same for W
    for i = 1:NCOLS
        mul!(Tcr, view(W3, :, :, i), Σ)
        for j = 1:NCOLS
            BLAS.axpy!(iXᵀX[i, j], Tcr, view(BT3, :, :, j))
        end
    end

    # C = W * B = W * (Bᵀ)ᵀ  -  Tcc contains C
    mul!(Tcc, W, transpose(BT))
    # ϰ = C \ resid  
    # note: since C is SPD, the fastest inverse is via Cholesky
    # cTcc = cholesky!(Symmetric(Tcc, :U)) 
    # Hmmm, actually cTcc may not be SPD, it may be S-semi-PD. 
    # QR with pivoting, not as fast, but should work in both cases.
    cTcc = qr!(Tcc, ColumnNorm())
    # Alternatively, try Cholesky and, if it fails, regularize away the zero eigenvalues
    # cTcc = cholesky!(Symmetric(Tcc, :U), check = false)
    # if cTcc.info != 0
    #     mul!(Tcc, W, transpose(BT))
    #     Tcc += √eps(one(T))*I(ncons)
    #     cTcc = cholesky!(Symmetric(Tcc, :U))
    # end
    # ϰ is computed in-place: Tc contains resid before and ϰ after the next line
    ldiv!(cTcc, Tc)
    # M = M + B * ϰ
    mul!(vec(M), transpose(BT), Tc, 1.0, 1.0)

    return M
end


