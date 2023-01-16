##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

# https://netlib.org/lapack/lug/node56.html
# https://netlib.org/lapack/lug/node57.html
# https://netlib.org/lapack/lug/node58.html


import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS
import LinearAlgebra.LAPACK

struct QZData
    N::BlasInt
    S::Matrix{Float64}
    T::Matrix{Float64}
    Q::Matrix{Float64}
    Z::Matrix{Float64}
    α_re::Vector{Float64}
    α_im::Vector{Float64}
    β::Vector{Float64}
    work::Vector{Float64}
    lwork::Ref{BlasInt}
    bwork::Vector{BlasInt}
    lbwork::Ref{BlasInt}
    sdim::Ref{BlasInt}
    info::Ref{BlasInt}
end

# reuse the space allocated in bwork for iwork -- in Fortran INTEGER and LOGICAL are both 32-bit.
# CAREFUL not to use both in the same call to lapack. 
# bwork is used in call to dgges, iwork is used in call to dtgsen
Base.getproperty(qz::QZData, name::Symbol) = getfield(qz, name === :iwork ? :bwork :
                                                          name === :liwork ? :lbwork :
                                                          name === :M ? :sdim :
                                                          name)

function QZData(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    (N, M) = size(A)
    @assert N == M && size(B) == (N, N)
    S = copyto!(similar(A), A)
    T = copyto!(similar(B), B)
    Q = similar(A)
    Z = similar(A)
    α_re = similar(A, N)
    α_im = similar(A, N)
    β = similar(A, N)
    work = Vector{Float64}(undef, 1)
    bwork = Vector{BlasInt}(undef, 0)
    return QZData(N, S, T, Q, Z, α_re, α_im, β, work, Ref(1), bwork, Ref(0), Ref(0), Ref(0))
end

function _dgges!(qz::QZData, selctg=C_NULL)
    # https://netlib.org/lapack/explore-html/d9/d8e/group__double_g_eeigen_ga8637d4b822e19d10327ddcb4235dc08e.html#ga8637d4b822e19d10327ddcb4235dc08e
    if selctg == C_NULL
        sort = 'N'
    else
        sort = 'S'
        resize!(qz.bwork, max(length(qz.bwork), qz.N))
        qz.lbwork[] = length(qz.bwork)
    end
    ccall((LAPACK.@blasfunc(dgges_), LAPACK.libblastrampoline), Cvoid, (
            Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},   # JOBVSL, JOBVSR, SORT, SELCTG
            Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, # N, in A out S, LDA, 
            Ptr{Float64}, Ref{BlasInt}, # in B out T, LDB, 
            Ptr{BlasInt}, # out SDIM
            Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, # ALPHAR, ALPHAI, BETA
            Ptr{Float64}, Ref{BlasInt}, # VSL, LDVSL
            Ptr{Float64}, Ref{BlasInt}, # VSR, LDVSR
            Ptr{Float64}, Ref{BlasInt}, # WORK, LWORK
            Ptr{BlasInt}, Ref{BlasInt}, # BWORK, INFO
        ),
        'V', 'V', sort, selctg,
        qz.N, qz.S, BlasInt(max(1, stride(qz.S, 2))),
        qz.T, BlasInt(max(1, stride(qz.T, 2))),
        qz.sdim,
        qz.α_re, qz.α_im, qz.β,
        qz.Q, BlasInt(max(1, stride(qz.Q, 2))),
        qz.Z, BlasInt(max(1, stride(qz.Z, 2))),
        qz.work, qz.lwork, qz.bwork, qz.info
    )
    return qz.info[]
end

function call_dgges!(qz)
    # first call to ask how much work space it needs 
    qz.lwork[] = -1
    info = _dgges!(qz)
    if info != 0
        error("Call to dgges failed with INFO = $(qz.info[])" * qz.info[] > qz.N ? " (N=$(qz.N))." : ".")
    end
    # allocate work space and call again
    qz.lwork[] = BlasInt(qz.work[1])
    resize!(qz.work, qz.lwork[])
    info = _dgges!(qz)
    if info != 0
        error("Call to dgges failed with INFO = $(qz.info[])" * qz.info[] > qz.N ? " (N=$(qz.N))." : ".")
    end
    return nothing
end

function _dtgsen!(qz::QZData, select::Vector{BlasInt})
    # https://netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gaba8441d4f7374bbcf6c093dbec0b517e.html#gaba8441d4f7374bbcf6c093dbec0b517e
    _zero = BlasInt(0)
    _one = BlasInt(1)
    ccall((LAPACK.@blasfunc(dtgsen_), LAPACK.libblastrampoline), Cvoid, (
            Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, # IJOB, WANTQ, WANTZ,
            Ptr{BlasInt}, # SELECT
            Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, # N, in/out A (a.k.a. S), LDA, 
            Ptr{Float64}, Ref{BlasInt}, # in/out B (a.k.a. T), LDB, 
            Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, # ALPHAR, ALPHAI, BETA
            Ptr{Float64}, Ref{BlasInt}, # VSL, LDVSL
            Ptr{Float64}, Ref{BlasInt}, # VSR, LDVSR
            Ptr{BlasInt}, # out M
            Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, # PL, PR, DIF
            Ptr{Float64}, Ref{BlasInt}, # WORK, LWORK
            Ptr{BlasInt}, Ref{BlasInt}, # IWORK, LIWORK
            Ptr{BlasInt}, # INFO
        ),
        _zero, _one, _one,
        select,
        qz.N, qz.S, BlasInt(max(1, stride(qz.S, 2))),
        qz.T, BlasInt(max(1, stride(qz.T, 2))),
        qz.α_re, qz.α_im, qz.β,
        qz.Q, BlasInt(max(1, stride(qz.Q, 2))),
        qz.Z, BlasInt(max(1, stride(qz.Z, 2))),
        qz.M,
        Ptr{Float64}(), Ptr{Float64}(), Ptr{Float64}(),
        qz.work, qz.lwork,
        qz.iwork, qz.liwork,
        qz.info
    )
    return qz.info[]
end

function call_dtgsen!(qz, select)
    # first call to figure out how much work space is needed 
    qz.lwork[] = -1
    resize!(qz.work, max(length(qz.work), 1))
    qz.liwork[] = -1
    resize!(qz.iwork, max(length(qz.iwork), 1))
    info = _dtgsen!(qz, select)
    if info != 0
        error("Call to dtgsen failed with INFO = $(qz.info[]).")
    end
    # resize work arrays, if necessary, and call again, this time for real 
    resize!(qz.work, max(length(qz.work), Int(qz.work[1])))
    qz.lwork[] = length(qz.work)
    resize!(qz.iwork, max(length(qz.iwork), Int(qz.iwork[1])))
    qz.liwork[] = length(qz.iwork)
    info = _dtgsen!(qz, select)
    if info != 0
        error("Call to dtgsen failed with INFO = $(qz.info[]).")
    end
    return nothing
end

"returns >0 for stable and <0 for unstable eigenvalues"
_diffg(qz) = @.(qz.α_re^2 + qz.α_im^2 - qz.β^2)

# macro _select_stable(cutoff::Float64=0.0)
#     return quote
#         @cfunction(function (a_re, a_im, b)
#                 return BlasInt(unsafe_load(a_re)^2 + unsafe_load(a_im)^2 - unsafe_load(b)^2 > $cutoff)
#             end,
#             BlasInt, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64})
#         )
#     end |> esc
# end

"""
    run_qz(A, B [, nstable])

Compute the QZ factorization (a.k.a. Generalized Schur decomposition) of the
matrix pencil (A,B). Return an instance of `QZData`, which contains the `Q,S,T,Z` 
matrices of the factorization.

The optional argument `nstable` is an integer that controls whether or not to
sort the eigenvalues in the factorization. If `nstable < 0` (default), there is
no sorting. If `nstable = 0` the eigenvalues are sorted into those >1 in the
top-left and those ≤0 in the bottom-right. If `nstable > 0`, then the largest
(in absolute value) `nstable` eigenvalues go in the top-left block and the rest
go in the bottom-right block.
"""
function run_qz(A::Matrix, B::Matrix, want_stable::Int=-1)
    qz = QZData(A, B)
    _run_qz!(qz, want_stable)
    return qz
end

function _run_qz!(qz::QZData, want_stable::Int)
    call_dgges!(qz)
    want_stable < 0 && return qz
    # we need sorting
    #           _dgges!(qz, @_select_stable)
    # doesn't work due to round-off errors: there may be unit eigenvalues that are ±eps()
    # 
    # instead, find a "cutoff value" in the vector of eigenvalues, such that we have
    # `want_stable` of them strictly above the cutoff and the rest below or at the cutoff.
    N = qz.N
    want_stable >= N && return qz
    if qz.lwork[] < N
        resize!(qz.work, N)
        qz.lwork[] = N
    end
    copyto!(qz.work, _diffg(qz))
    cutoff = want_stable == 0 ? 0.0 : partialsort(qz.work[1:N], N - want_stable)
    select = BlasInt[w > cutoff for w in qz.work[1:N]]
    call_dtgsen!(qz, select)
    return qz
end


"""
    run_qz!(qz, A, B, [nstable])

Like [`run_qz`](@ref) but reuses the given `qz` data structure. Dimensions of
`A` and `B` must be `(qz.N, qz.N)`.
"""
function run_qz!(qz::QZData, A::Matrix{Float64}, B::Matrix{Float64}, want_stable::Int=-1)
    if size(A) == size(B) == (qz.N, qz.N)
        nothing
    else
        throw(ArgumentError("Matrices of incompatible sizes."))
    end
    copyto!(qz.S, A)
    copyto!(qz.T, B)
    _run_qz!(qz, want_stable)
    return qz
end
