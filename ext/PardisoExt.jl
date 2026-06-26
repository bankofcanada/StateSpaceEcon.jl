##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

module PardisoExt

# StateSpaceEcon <-> Pardiso bridge.
#
# Pardiso is consumed as an optional dependency via this package
# extension. Without Pardiso loaded, the solver uses Julia's default
# UMFPACK path (`J \ R`). With Pardiso loaded and the user passing
# `linsolve=:pardiso` to `simulate!`/`plan_simulate!`, the inner Newton
# step routes through MKLPardiso: symbolic factorization once, numeric
# refactor + iterative-refinement solve per Newton iteration, and a
# `RELEASE_ALL` finalizer at the end of the Newton loop.

using StateSpaceEcon
import StateSpaceEcon.StackedTimeSolver: _solve_jacobian, _init_linsolve,
                                         _finalize_linsolve!
using SparseArrays: SparseMatrixCSC
using Pardiso

# Mutable so we can attach a finalizer and track whether the symbolic
# factorization has been run.
mutable struct PardisoFactorization
    ps::MKLPardisoSolver
    analyzed::Bool
    released::Bool
    # last J handed to Pardiso, in whatever permutation Pardiso wants
    Jp::Union{Nothing, SparseMatrixCSC{Float64, Int32}}
end

function _new_pardiso_state()
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_NONSYM)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    # iparm(2) = 2 -> parallel (OpenMP) nested-dissection reordering. See
    # the Intel oneMKL Pardiso iparm reference.
    set_iparm!(ps, 2, 2)
    state = PardisoFactorization(ps, false, false, nothing)
    finalizer(_release_pardiso!, state)
    return state
end

function _release_pardiso!(state::PardisoFactorization)
    state.released && return
    state.released = true
    try
        set_phase!(state.ps, Pardiso.RELEASE_ALL)
        pardiso(state.ps)
    catch
        # finalizers must not throw
    end
    return
end

# --- StackedTimeSolver hooks ------------------------------------------

_init_linsolve(::Val{:pardiso}) = _new_pardiso_state()

function _finalize_linsolve!(::Val{:pardiso}, state::PardisoFactorization)
    _release_pardiso!(state)
    return
end
# Be defensive against alternate paths returning nothing.
_finalize_linsolve!(::Val{:pardiso}, ::Nothing) = nothing

function _solve_jacobian(::Val{:pardiso}, J::SparseMatrixCSC,
                          R::AbstractVector, state::PardisoFactorization)
    ps = state.ps
    # Pardiso requires Int32 indices and a particular layout for REAL_NONSYM;
    # `get_matrix` normalises whatever sparse matrix we hand it.
    Jp = get_matrix(ps, J, :N)
    state.Jp = Jp
    # Symbolic factorization once (pattern is fixed across Newton iters).
    if !state.analyzed
        set_phase!(ps, Pardiso.ANALYSIS)
        pardiso(ps, Jp, Float64[])
        state.analyzed = true
    end
    # Numeric refactor per call (Jacobian values change each Newton step).
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, Jp, Float64[])
    # Solve with iterative refinement.
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Δ = similar(Vector{Float64}(undef, length(R)))
    pardiso(ps, Δ, Jp, Vector{Float64}(R))
    return Δ, state
end

end # module PardisoExt
