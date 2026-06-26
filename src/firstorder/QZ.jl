##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Generalised Schur (QZ) decomposition for the first-order solver.
#
# We use the high-level `LinearAlgebra.schur` / `ordschur`, which wrap the
# LAPACK `dgges_` (decompose) and `dtgsen_` (reorder) routines.
#
# For a matrix pencil (A, B) the QZ factorization is
#     A = Q S Z'        B = Q T Z'
# with Q, Z orthogonal and S, T (quasi-)upper-triangular. The generalized
# eigenvalues are λ = α / β, with α = diag-ish of S (complex, capturing
# 2x2 blocks for complex pairs) and β = diag of T. `LinearAlgebra.schur`
# exposes S, T, Q (= LAPACK VSL), Z (= LAPACK VSR), α (complex) and β.
# ----------------------------------------------------------------------

"""
Result of a (possibly reordered) generalised Schur decomposition of the
pencil `(A, B)`: `A = Q*S*Z'`, `B = Q*T*Z'`.
"""
struct QZResult
    N::Int
    S::Matrix{Float64}
    T::Matrix{Float64}
    Q::Matrix{Float64}   # = LAPACK VSL (used as Q'EX)
    Z::Matrix{Float64}   # = LAPACK VSR
    α_re::Vector{Float64}
    α_im::Vector{Float64}
    β::Vector{Float64}
end

# `_diffg`: >0 for "stable" (|λ|>1 in this proxy), <0 otherwise.
# Note the proxy is α_re² + α_im² - β², i.e. compares |α|² vs |β|², which
# sorts by the generalized eigenvalue magnitude |λ|² = |α|²/β².
_diffg(qz::QZResult) = @. qz.α_re^2 + qz.α_im^2 - qz.β^2

"""
    run_qz(A, B, want_stable=-1) -> QZResult

Compute the QZ factorization of the pencil `(A, B)`. When `want_stable < 0`
no reordering is done. When `want_stable >= 0`, the eigenvalues are sorted
so the `want_stable` with the largest `|α|² - β²` (i.e. largest magnitude)
go into the top-left block and the rest into the bottom-right.

`want_stable == 0` sorts strictly by sign of the proxy (cutoff at 0); a
positive count finds the order statistic so exactly `want_stable` land in
the top-left.
"""
function run_qz(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64},
                want_stable::Int=-1)
    N = LinearAlgebra.checksquare(A)
    LinearAlgebra.checksquare(B) == N ||
        throw(ArgumentError("run_qz: A and B must be the same square size"))

    F = LinearAlgebra.schur(Matrix{Float64}(A), Matrix{Float64}(B))
    qz = QZResult(N, F.S, F.T, F.Q, F.Z,
                  real.(F.α), imag.(F.α), F.β)

    want_stable < 0 && return qz
    want_stable >= N && return qz

    # Find a cutoff in the eigenvalue proxy so that exactly `want_stable`
    # entries are strictly above it (-> top-left). Using a cutoff rather
    # than the LAPACK select callback avoids round-off flips on unit
    # eigenvalues.
    diffg = _diffg(qz)
    cutoff = want_stable == 0 ? 0.0 : partialsort(copy(diffg), N - want_stable)
    select = [w > cutoff for w in diffg]

    F2 = LinearAlgebra.ordschur(F, select)
    return QZResult(N, F2.S, F2.T, F2.Q, F2.Z,
                    real.(F2.α), imag.(F2.α), F2.β)
end
