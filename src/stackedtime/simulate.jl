##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Boundary clamping helpers
# ----------------------------------------------------------------------
#
# `x_full` is a (T + maxlag + maxlead) x n_var matrix; row `t+maxlag` is
# simulation period t. Initial conditions live in rows 1..maxlag and are
# fixed. Terminal conditions live in rows maxlag+T+1..end and are fixed
# (typically set to steady state). The "unknowns" are the rows
# maxlag+1..maxlag+T, which we view-into as a length-`T*n_var` vector.

@inline _row(t::Int, plan::StackedTimePlan) = t + plan.maxlag

# ----------------------------------------------------------------------
# Residual + Jacobian assembly
# ----------------------------------------------------------------------

"""
    stacked_residual!(R, x_full, e_full, plan) -> R

Compute the global stacked residual vector. `x_full` and `e_full` are
the (T+maxlag+maxlead)xn matrices including boundary rows. `R` has
length `T * n_eq`.
"""
function stacked_residual!(R::AbstractVector{Float64},
                            x_full::AbstractMatrix{Float64},
                            e_full::AbstractMatrix{Float64},
                            plan::StackedTimePlan)
    p = plan.param_values
    T = plan.T
    n_eq = plan.n_eq
    @inbounds for t in 1:T
        for i in 1:n_eq
            eqn = plan.model[i]
            sm = plan.slot_maps[i]
            x_eqn = _gather_x(x_full, e_full, sm, t, plan)
            R[(t - 1) * n_eq + i] = eqn.eval_resid(x_eqn, p)
        end
    end
    return R
end

"""
    stacked_RJ!(R, J, x_full, e_full, plan) -> (R, J)

Compute residual and refresh the sparse Jacobian's nzvals. `J` must be
`plan.J` (the prebuilt sparsity pattern); we mutate its nzval in place.
"""
function stacked_RJ!(R::AbstractVector{Float64},
                      x_full::AbstractMatrix{Float64},
                      e_full::AbstractMatrix{Float64},
                      plan::StackedTimePlan)
    p = plan.param_values
    T = plan.T
    n_eq = plan.n_eq
    nz = nonzeros(plan.J)
    fill!(nz, 0.0)
    grad_buf = Vector{Float64}(undef, 0)
    @inbounds for t in 1:T
        for i in 1:n_eq
            eqn = plan.model[i]
            sm = plan.slot_maps[i]
            if length(grad_buf) != eqn.n_x
                resize!(grad_buf, eqn.n_x)
            end
            x_eqn = _gather_x(x_full, e_full, sm, t, plan)
            R[(t - 1) * n_eq + i] = eqn.eval_RJ!(grad_buf, x_eqn, p)
            block = (t - 1) * n_eq + i
            bi = plan.BI[block]
            for k in 1:length(bi)
                nz_idx = bi[k]
                nz_idx == 0 && continue              # boundary slot
                nz[nz_idx] += grad_buf[k]
            end
        end
    end
    return R, plan.J
end

# Build x_eqn from x_full / e_full according to slot map.
@inline function _gather_x(x_full::AbstractMatrix{Float64},
                            e_full::AbstractMatrix{Float64},
                            sm::StackedSlotMap, t::Int,
                            plan::StackedTimePlan)
    n = length(sm)
    x_eqn = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        kind, idx, off = sm[k]
        row = t + off + plan.maxlag
        x_eqn[k] = kind === :var ? x_full[row, idx] : e_full[row, idx]
    end
    return x_eqn
end

# ----------------------------------------------------------------------
# Newton solver
# ----------------------------------------------------------------------

"""
    simulate!(plan; x_init, x_term, e_full, x_guess=nothing,
              tol=1e-10, maxiter=50, verbose=false)
        -> (x_full, converged, iters)

Solve the stacked-time system. Inputs:
- `x_init :: Matrix{Float64}` - `maxlag x n_var` initial conditions
- `x_term :: Matrix{Float64}` - `maxlead x n_var` terminal conditions (typically ss)
- `e_full :: Matrix{Float64}` - `(T + maxlag + maxlead) x n_shock` exogenous shocks
- `x_guess :: Matrix{Float64}` (optional) - `T x n_var` starting guess for unknowns

Returns `x_full` of size `(T + maxlag + maxlead) x n_var` with the boundary
rows filled in and the interior solved.
"""
function simulate!(plan::StackedTimePlan;
                   x_init::AbstractMatrix{Float64},
                   x_term::AbstractMatrix{Float64},
                   e_full::AbstractMatrix{Float64},
                   x_guess::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                   tol::Float64 = 1e-10,
                   maxiter::Int = 50,
                   verbose::Bool = false,
                   linsolve::Symbol = :umfpack)
    T, n_var = plan.T, plan.n_var
    maxlag, maxlead = plan.maxlag, plan.maxlead
    size(x_init) == (maxlag, n_var) ||
        error("x_init must be $(maxlag)x$(n_var)")
    size(x_term) == (maxlead, n_var) ||
        error("x_term must be $(maxlead)x$(n_var)")
    size(e_full) == (T + maxlag + maxlead, plan.n_shock) ||
        error("e_full must be $(T+maxlag+maxlead)x$(plan.n_shock)")

    x_full = zeros(T + maxlag + maxlead, n_var)
    if maxlag > 0
        x_full[1:maxlag, :] .= x_init
    end
    if maxlead > 0
        x_full[maxlag+T+1:end, :] .= x_term
    end
    if x_guess !== nothing
        size(x_guess) == (T, n_var) ||
            error("x_guess must be $(T)x$(n_var)")
        x_full[maxlag+1:maxlag+T, :] .= x_guess
    else
        # Default guess: linear interpolation initial -> terminal.
        for v in 1:n_var
            a = maxlag > 0 ? x_init[end, v] : (maxlead > 0 ? x_term[1, v] : 0.0)
            b = maxlead > 0 ? x_term[1, v] : a
            for t in 1:T
                x_full[maxlag + t, v] = a + (b - a) * (t / (T + 1))
            end
        end
    end

    R = Vector{Float64}(undef, T * plan.n_eq)
    iter = 0
    converged = false
    lv = Val(linsolve)
    lstate = _init_linsolve(lv)
    try
        while iter < maxiter
            iter += 1
            stacked_RJ!(R, x_full, e_full, plan)
            rnorm = norm(R, Inf)
            verbose && @info "stacked iter $iter" rnorm
            if rnorm < tol
                converged = true
                break
            end
            Δ, lstate = _solve_jacobian(lv, plan.J, R, lstate)
            @inbounds for v in 1:n_var, t in 1:T
                x_full[maxlag + t, v] -= Δ[_xidx(t, v, T)]
            end
            if norm(Δ, Inf) < tol
                stacked_residual!(R, x_full, e_full, plan)
                converged = norm(R, Inf) < tol
                break
            end
        end
    finally
        _finalize_linsolve!(lv, lstate)
    end
    return x_full, converged, iter
end
