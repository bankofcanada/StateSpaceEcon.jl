##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################


# selected sparse linear algebra library is a Symbol
const sf_libs = (
    :none,      # do not pre-factorize the Jacobian matrix
    :default,   # use the current default, which can be changed via use_umfpack() and use_pardiso()
    :umfpack,   # use Julia's standard library (UMFPACK)
    :pardiso,   # use Pardiso - the one included with MKL
)

global sf_default = :umfpack

"""
    use_umfpack()

Set the default sparse factorization library to UMFPACK (the one used in Julia's
standard library). See also [`use_pardiso`](@ref).

"""
@inline use_umfpack() = (global sf_default = :umfpack; nothing)

"""
    use_umfpack!(model)

Instruct the stacked-time solver to use Pardiso with this model. See also
[`use_pardiso!`](@ref).
"""
@inline use_umfpack!(m::Model) = (m.options.factorization = :umfpack; m.options)
export use_umfpack, use_umfpack!

"""
    use_pardiso()

Set the default sparse factorization library to Pardiso. See also
[`use_umfpack`](@ref).
"""
@inline use_pardiso() = (global sf_default = :pardiso; nothing)

"""
    use_pardiso!(model)

Instruct the stacked-time solver to use Pardiso with this model. 
"""
@inline use_pardiso!(m::Model) = (m.options.factorization = :pardiso; m.options)
export use_pardiso, use_pardiso!

### API

# a function to initialize a Factorization instance
# this is also a good place to do the symbolic analysis
# sf_prepare(A::SparseMatrixCSC, sparse_lib::Symbol=:default) = sf_prepare(Val(sparse_lib), A)
# sf_prepare(::Val{S}, args...) where {S} = throw(ArgumentError("Unknown sparse library $S. Try one of $(sf_libs)."))

# a function to calculate the numerical factors
# sf_factor!(f::Factorization, A::SparseMatrixCSC) = throw(ArgumentError("Unknown factorization type $(typeof(f))."))

# a function to solve the linear system
# sf_solve!(f::Factorization, x::AbstractArray) = throw(ArgumentError("Unknown factorization type $(typeof(f))."))


###########################################################################
### :none 

abstract type SF_Factorization{Tv} <: Factorization{Tv} end

mutable struct NoFactorization{Tv} <: SF_Factorization{Tv}
    A::SparseMatrixCSC{Tv,Int}
end
# don't factorize, just store the matrix
sf_prepare(::Val{:none}, A::SparseMatrixCSC) = NoFactorization{Float64}(A)
# don't factorize, just store the matrix
sf_factor!(f::NoFactorization, A::SparseMatrixCSC) = (f.A = A; f)
# we could do ldiv(f.A, x), but we insist on throwing an error -- if you want to solve, don't use model.factorization = :none
sf_solve!(f::NoFactorization, x::AbstractArray) = error("Cannot solve without valid factorization.")

###########################################################################
###  Default (UMFPACK)
sf_prepare(::Val{:default}, A::SparseMatrixCSC) = sf_prepare(Val(sf_default), A)

function _sf_same_sparse_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return (A.m == B.m) && (A.n == B.n) && (A.colptr == B.colptr) && (A.rowval == B.rowval)
end

macro _sf_check_factorize(exception, expression)
    error = gensym("error")
    return esc(quote
        try
            $expression
        catch $error
            if $error isa $exception
                @error("The system is underdetermined with the given set of equations and final conditions.")
            end
            rethrow()
        end
    end)
end

mutable struct LUFactorization{Tv<:Real} <: SF_Factorization{Tv}
    F::SuiteSparse.UMFPACK.UmfpackLU{Tv,Int}
    A::SparseMatrixCSC{Tv,Int}
end

@timeit_debug timer "sf_prepare_lu" function sf_prepare(::Val{:umfpack}, A::SparseMatrixCSC)
    Tv = eltype(A)
    F = @_sf_check_factorize(SingularException, @timeit_debug timer "_lu_full" lu(A))
    return LUFactorization{Tv}(F, A)
end

@timeit_debug timer "sf_factor!_lu" function sf_factor!(f::LUFactorization, A::SparseMatrixCSC)
    _A = f.A
    if _sf_same_sparse_pattern(A, _A)
        if A.nzval ≈ _A.nzval
            # matrix hasn't changed significantly
            nothing
        else
            # sparse pattern is the same, different numbers
            f.A = A
            @_sf_check_factorize(SingularException, @timeit_debug timer "_lu_num" lu!(f.F, A))
        end
    else
        # totally new matrix, start over
        f.A = A
        f.F = @_sf_check_factorize(SingularException, @timeit_debug timer "_lu_full" lu(A))
    end
    return f
end

@timeit_debug timer "sf_solve!_lu" sf_solve!(f::LUFactorization, x::AbstractArray) = (ldiv!(f.F, x); x)

###########################################################################
###  Pardiso (thanks to @KristofferC)

# See https://github.com/JuliaSparse/Pardiso.jl/blob/master/examples/exampleunsym.jl
# See https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html

mutable struct PardisoFactorization{Tv<:Real} <: SF_Factorization{Tv}
    ps::MKLPardisoSolver
    A::SparseMatrixCSC{Tv,Int}
end

@timeit_debug timer "sf_prepare_par" function sf_prepare(::Val{:pardiso}, A::SparseMatrixCSC)
    Tv = eltype(A)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_NONSYM)
    pardisoinit(ps)
    # set_msglvl!(ps, 1) # make pardiso verbose
    fix_iparm!(ps, :N)
    set_iparm!(ps, 2, 3) # Select algorithm
    # set_iparm!(ps, 8, 5) # Maximum number of iterative refinement steps
    set_iparm!(ps, 10, 15) # Pivot perturbation (if pivot is less than 10^(-iparam[10]))
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
    @_sf_check_factorize Union{Pardiso.PardisoException,Pardiso.PardisoPosDefException} begin
        set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
        pardiso(ps, pf.A, Float64[])
        # Fail if Pardiso perturbed any pivots
        get_iparm(ps, 14) == 0 || throw(Pardiso.PardisoException("Zero or near-zero pivot."))
    end
    return pf
end

@timeit_debug timer "_pardso_num" function _pardiso_numeric!(pf::PardisoFactorization)
    # run the analysis phase
    ps = pf.ps
    @_sf_check_factorize Union{Pardiso.PardisoException,Pardiso.PardisoPosDefException} begin
        set_phase!(ps, Pardiso.NUM_FACT)
        pardiso(ps, pf.A, Float64[])
        # Fail if Pardiso perturbed any pivots
        get_iparm(ps, 14) == 0 || throw(Pardiso.PardisoException("Zero or near-zero pivot."))
    end
    return pf
end

@timeit_debug timer "sf_factor!_par" function sf_factor!(pf::PardisoFactorization, A::SparseMatrixCSC)
    A = get_matrix(pf.ps, A, :N)::typeof(A)
    _A = pf.A
    if _sf_same_sparse_pattern(A, _A)
        if A.nzval ≈ _A.nzval
            # same matrix, factorization hasn't changed
            nothing
        else
            # same sparsity pattern, but different numbers
            pf.A = A
            _pardiso_numeric!(pf)
        end
    else
        # totally new matrix, start over
        pf.A = A
        _pardiso_full!(pf)
    end
    return pf
end

@timeit_debug timer "sf_solve!_par" function sf_solve!(pf::PardisoFactorization, x::AbstractArray)
    ps = pf.ps
    @_sf_check_factorize Union{Pardiso.PardisoException,Pardiso.PardisoPosDefException} begin
        set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
        pardiso(ps, x, pf.A, copy(x))
    end
    return x
end

