##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Pluggable sparse linear solve.
#
# `_solve_jacobian(::Val{linsolve}, J, R, state)` solves `J Δ = R` and
# returns `(Δ, state)`. The default `:umfpack` path delegates to Julia's
# `\` and uses no state. The `:pardiso` path is provided by the
# PardisoExt extension; it threads a factorization state across Newton
# iterations so the symbolic factorization is reused.
#
# `_init_linsolve(::Val{linsolve})` returns the initial state value
# (`nothing` by default); `_finalize_linsolve!(::Val{linsolve}, state)`
# releases solver resources at the end of the Newton loop.
# ----------------------------------------------------------------------

function _solve_jacobian end
function _init_linsolve end
function _finalize_linsolve! end

@inline _init_linsolve(::Val) = nothing
@inline _finalize_linsolve!(::Val, _) = nothing

@inline function _solve_jacobian(::Val{:umfpack}, J::SparseMatrixCSC,
                                  R::AbstractVector, state)
    return J \ R, state
end

function _solve_jacobian(::Val{ls}, J::SparseMatrixCSC,
                         R::AbstractVector, state) where {ls}
    if ls === :pardiso
        error("simulate!: linsolve=:pardiso requested but the Pardiso " *
              "extension is not loaded. Run `using Pardiso` before " *
              "calling simulate!/plan_simulate! with linsolve=:pardiso.")
    else
        error("simulate!: unknown linsolve=$(repr(ls)). " *
              "Supported: :umfpack (default), :pardiso (requires `using Pardiso`).")
    end
end

# ----------------------------------------------------------------------
# Sparsity construction
# ----------------------------------------------------------------------

# Build the sparse Jacobian and BI scatter map.
#
# Rows: `(t-1)*n_eq + i` for equation i at simulation period t (t in 1..T).
# Cols: `_xidx(t', v)` for unknown (t', v) where t' is in 1..T.
#
# References to t' < 1 (initial conditions) or t' > T (terminal) are
# *not* unknowns - their derivatives don't appear in J. We still
# include the residual contribution (initial/terminal x values are
# clamped on each evaluation).
function _build_sparsity(T::Int, n_var::Int, n_eq::Int,
                          slot_maps::Vector{StackedSlotMap})
    # First pass: collect (row, col) pairs and remember, per (t, eq, slot_k),
    # the position in the COO list - so after building J via sparse(), we
    # can map slot_k -> nzval index.
    Is = Int[]
    Js = Int[]
    # For each (t, eq) block: vector of (slot_k, coo_index_or_zero). zero
    # means "boundary slot, no Jacobian contribution".
    block_slots = Vector{Vector{Tuple{Int, Int}}}()
    sizehint!(block_slots, T * n_eq)

    for t in 1:T
        for i in 1:n_eq
            sm = slot_maps[i]
            slots = Vector{Tuple{Int, Int}}()
            for (k, entry) in pairs(sm)
                kind, idx, off = entry
                if kind === :var
                    tprime = t + off
                    if 1 <= tprime <= T
                        push!(Is, (t - 1) * n_eq + i)
                        push!(Js, _xidx(tprime, idx, T))
                        push!(slots, (k, length(Is)))
                    else
                        push!(slots, (k, 0))   # boundary
                    end
                else
                    # Shock: never an unknown.
                    push!(slots, (k, 0))
                end
            end
            push!(block_slots, slots)
        end
    end

    # Build sparse matrix; sparse() sums duplicates and orders by (col, row).
    n_rows = T * n_eq
    n_cols = T * n_var
    Vs = ones(Float64, length(Is))           # placeholder values
    J = sparse(Is, Js, Vs, n_rows, n_cols)

    # Now invert: for each COO index, find its position in J.nzval.
    # Build a dict (row, col) -> nzval index.
    nz_index = Dict{Tuple{Int, Int}, Int}()
    sizehint!(nz_index, length(Is))
    rows = rowvals(J)
    @inbounds for col in 1:n_cols
        for p in nzrange(J, col)
            nz_index[(rows[p], col)] = p
        end
    end

    # For each block, build BI[b] = vector of nzval-indices in tsref order
    # (length = length(slot_maps[i]); 0 marks boundary slots - we skip
    # those when scattering).
    BI = Vector{Vector{Int}}(undef, T * n_eq)
    for (b, slots) in pairs(block_slots)
        n_slots = isempty(slots) ? 0 : maximum(s -> s[1], slots)
        bi = zeros(Int, n_slots)
        for (k, coo_idx) in slots
            if coo_idx == 0
                bi[k] = 0
            else
                # Recover (row, col) from Is/Js at coo_idx.
                bi[k] = nz_index[(Is[coo_idx], Js[coo_idx])]
            end
        end
        BI[b] = bi
    end

    # Reset nzval - sparse() set them to 1.0 from our placeholder.
    fill!(nonzeros(J), 0.0)

    return J, BI
end
