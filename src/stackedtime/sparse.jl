##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################


# using LinearAlgebra
# using SparseArrays
# using SuiteSparse
# using SuiteSparse.UMFPACK
# using Pardiso


### API

# selected sparse linear algebra library is a Symbol
const sf_libs = (
    :default,   # use Julia's standard library (UMFPACK)
    :lu,  # same as :default
    # :qr,  # use qr-decomposition, rather than lu (still UMFPACK)
    :pardiso,   # use Pardiso - the one included with MKL
)

global sf_default = :lu

# a function to initialize a Factorization instance
#  this is also a good place to do the symbolic analysis
sf_prepare(A::SparseMatrixCSC, sparse_lib::Symbol=:default) = sf_prepare(Val(sparse_lib), A)
sf_prepare(::Val{S}, args...) where {S} = throw(ArgumentError("Unknown sparse library $S. Try one of $(sf_libs)."))

# a function to calculate the numerical factors
sf_factor!(f::Factorization, A::SparseMatrixCSC) = throw(ArgumentError("Unknown factorization type $(typeof(f))."))

# a function to solve the linear system
sf_solve!(f::Factorization, x::AbstractArray) = throw(ArgumentError("Unknown factorization type $(typeof(f))."))


###  Default (UMFPACK)
sf_prepare(::Val{:default}, A::SparseMatrixCSC) = sf_prepare(Val(sf_default), A)

@timeit_debug timer "sf_prepare_lu" sf_prepare(::Val{:lu}, A::SparseMatrixCSC) = lu(A)
@timeit_debug timer "sf_factor!_lu" sf_factor!(f::SuiteSparse.UMFPACK.UmfpackLU, A::SparseMatrixCSC) = (lu!(f, A); f)
@timeit_debug timer "sf_solve!_lu" sf_solve!(f::SuiteSparse.UMFPACK.UmfpackLU, x::AbstractArray) = (ldiv!(f, x); x)

# @timeit_debug timer "sf_prepare_qr" sf_prepare(::Val{:qr}, A::SparseMatrixCSC) = qr(A)
# @timeit_debug timer "sf_factor!_qr" sf_factor!(f::SuiteSparse.SPQR.QRSparse, A::SparseMatrixCSC) = (qr!(f, A); f)
# @timeit_debug timer "sf_solve!_qr" sf_solve!(f::SuiteSparse.SPQR.QRSparse, x::AbstractArray) = (ldiv!(f, x); x)

###  Pardiso (thanks to @KristofferC)

# See https://github.com/JuliaSparse/Pardiso.jl/blob/master/examples/exampleunsym.jl

mutable struct PardisoFactorization{Tv<:Real} <: Factorization{Tv}
    ps::MKLPardisoSolver
    A::SparseMatrixCSC{Tv,Int}
end

@timeit_debug timer "sf_prepare_par" function sf_prepare(::Val{:pardiso}, A::SparseMatrixCSC)
    Tv = eltype(A)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_NONSYM)
    pardisoinit(ps)
    # See https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html
    fix_iparm!(ps, :N)
    set_iparm!(ps, 2, 2) # The parallel (OpenMP) version of the nested dissection algorithm.
    pf = PardisoFactorization{Tv}(ps, get_matrix(ps, A, :N))
    finalizer(pf) do x
        set_phase!(x.ps, Pardiso.RELEASE_ALL)
        pardiso(x.ps)
    end
    _pardiso_full!(pf)
    return pf
end

@timeit_debug timer "_pardso_full" function _pardiso_full!(pf::PardisoFactorization)
    # run the analysis phase
    ps = pf.ps
    set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    pardiso(ps, pf.A, Float64[])
    return pf
end

@timeit_debug timer "_pardso_num" function _pardiso_numeric!(pf::PardisoFactorization)
    # run the analysis phase
    ps = pf.ps
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, pf.A, Float64[])
    return pf
end

@timeit_debug timer "sf_factor!_par" function sf_factor!(pf::PardisoFactorization, A::SparseMatrixCSC)
    A = get_matrix(pf.ps, A, :N)::typeof(A)
    # can we reuse the 
    _A = pf.A
    if A.m == _A.m && A.n == _A.n && A.colptr == _A.colptr && A.rowval == _A.rowval
        if A.nzval ≈ _A.nzval
            nothing
        else
            pf.A = A
            _pardiso_numeric!(pf)
        end
    else
        pf.A = A
        _pardiso_full!(pf)
    end
    return pf
end

@timeit_debug timer "sf_solve!_par" function sf_solve!(pf::PardisoFactorization, x::AbstractArray)
    ps = pf.ps
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    pardiso(ps, x, pf.A, copy(x))
    return x
end

