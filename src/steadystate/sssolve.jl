##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Residual + Jacobian assembly
# ----------------------------------------------------------------------

"""
    ss_residual!(R, x_ss, prob) -> R

Compute the global SS residual vector in-place.
"""
function ss_residual!(R::AbstractVector{Float64},
                       x_ss::AbstractVector{Float64},
                       prob::SteadyStateProblem)
    p = prob.param_values
    @inbounds for i in 1:prob.n_eq
        eqn = prob.model[i]
        sm = prob.slot_maps[i]
        x_eqn = _scatter_x(x_ss, sm, eqn.n_x)
        R[i] = eqn.eval_resid(x_eqn, p)
    end
    # User-supplied SS equations stacked below the auto-derived rows.
    @inbounds for j in 1:prob.n_ss_user
        sseq = prob.model.ss_eqns[j]
        sm = prob.ss_user_slot_maps[j]
        x_eqn = _scatter_x(x_ss, sm, sseq.eqn.n_x)
        R[prob.n_eq + j] = sseq.eqn.eval_resid(x_eqn, p)
    end
    return R
end

"""
    ss_RJ!(R, J, x_ss, prob) -> (R, J)

Compute residual and dense Jacobian (size `n_eq x n_var`) in-place.
The per-equation gradient is *summed* across all `tsrefs` slots that
correspond to the same variable, since at SS all time offsets collapse
to the same point.
"""
function ss_RJ!(R::AbstractVector{Float64},
                 J::AbstractMatrix{Float64},
                 x_ss::AbstractVector{Float64},
                 prob::SteadyStateProblem)
    p = prob.param_values
    fill!(J, 0.0)
    grad_buf = Vector{Float64}(undef, 0)
    @inbounds for i in 1:prob.n_eq
        eqn = prob.model[i]
        sm = prob.slot_maps[i]
        x_eqn = _scatter_x(x_ss, sm, eqn.n_x)
        if length(grad_buf) != eqn.n_x
            resize!(grad_buf, eqn.n_x)
        end
        R[i] = eqn.eval_RJ!(grad_buf, x_eqn, p)
        for (k, vidx) in pairs(sm)
            vidx === nothing && continue
            J[i, vidx] += grad_buf[k]
        end
    end
    # User-supplied SS equation rows.
    @inbounds for j in 1:prob.n_ss_user
        sseq = prob.model.ss_eqns[j]
        sm = prob.ss_user_slot_maps[j]
        x_eqn = _scatter_x(x_ss, sm, sseq.eqn.n_x)
        if length(grad_buf) != sseq.eqn.n_x
            resize!(grad_buf, sseq.eqn.n_x)
        end
        row = prob.n_eq + j
        R[row] = sseq.eqn.eval_RJ!(grad_buf, x_eqn, p)
        for (k, vidx) in pairs(sm)
            vidx === nothing && continue
            J[row, vidx] += grad_buf[k]
        end
    end
    return R, J
end

# Build the per-equation x vector by gathering from x_ss according to sm.
@inline function _scatter_x(x_ss::AbstractVector{Float64},
                             sm::SlotMap, n::Int)
    x_eqn = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        vidx = sm[k]
        x_eqn[k] = vidx === nothing ? 0.0 : x_ss[vidx]
    end
    return x_eqn
end

# ----------------------------------------------------------------------
# Newton solver - straight, no LM, no line search.
# ----------------------------------------------------------------------

"""
    sssolve!(prob; x0, tol=1e-10, maxiter=50, verbose=false,
             on_failure=:error)
        -> (x_ss, converged, iters)

Newton solve for the steady state. `x0` is the initial guess. Returns
the solution, a convergence flag, and the iteration count.

`on_failure` controls behaviour when the Newton iteration does not
reach `tol` within `maxiter`:

- `:error` (default) - the caller inspects the returned `converged`
  flag.
- `:diagnose` - on non-convergence, additionally return a
  [`SteadyStateDiagnosis`](@ref) of the last iterate as a fourth
  tuple element.
- `:warn` - `@warn` a printed diagnosis on non-convergence and
  return the same three-tuple as `:error`.
"""
function sssolve!(prob::SteadyStateProblem;
                  x0::AbstractVector{Float64} = ones(prob.n_var),
                  tol::Float64 = 1e-10,
                  maxiter::Int = 50,
                  verbose::Bool = false,
                  on_failure::Symbol = :error)
    on_failure in (:error, :diagnose, :warn) ||
        error("sssolve!: on_failure must be :error, :diagnose, or :warn (got :$on_failure)")
    n = prob.n_var
    m = n_total_eq(prob)
    overdet = m > n
    x = collect(Float64, x0)
    R = Vector{Float64}(undef, m)
    J = Matrix{Float64}(undef, m, n)

    iter = 0
    converged = false
    while iter < maxiter
        iter += 1
        ss_RJ!(R, J, x, prob)
        rnorm = norm(R, Inf)
        verbose && @info "ss iter $iter" rnorm
        if rnorm < tol
            converged = true
            break
        end
        # Solve J Δ = R, then x <- x - Δ. Square: LU. Over-determined:
        # Gauss-Newton least-squares step via QR.
        # A singular Jacobian / rank-deficient QR means Newton has stalled
        # at a critical point; surface that as non-convergence so
        # `on_failure` can react.
        local Δ
        try
            if overdet
                Δ = qr(J, ColumnNorm()) \ R
            else
                F = lu(J)
                Δ = F \ R
            end
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            verbose && @info "ss iter $iter: singular Jacobian, halting"
            break
        end
        x .-= Δ
        if norm(Δ, Inf) < tol
            # Refresh residual to confirm.
            ss_residual!(R, x, prob)
            converged = norm(R, Inf) < tol
            break
        end
    end

    if !converged && on_failure === :diagnose
        return x, converged, iter, _diagnose_for_sssolve(prob, x)
    elseif !converged && on_failure === :warn
        @warn "sssolve! did not converge in $iter iterations\n" *
              sprint(show, _diagnose_for_sssolve(prob, x))
    end
    return x, converged, iter
end
