##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

function _sim_lm_step(
    x::AbstractArray{Float64},
    Δx::AbstractArray{Float64},
    sd::StackedTimeSolverData;
    lm_params=[0.1, 8.0, 0.0]
)

    lambda, nu, qual = lm_params

    R, FJ = stackedtime_RJ(x, x, sd; factorization=:none)
    J = FJ.A

    nx = length(R)
    JtR = J'R
    M = J'J
    # max is to make sure no zeros in Jd
    Jd = max.(1e-10, vec(sum(abs2, J; dims=1)))
    for i = 1:nx
        @inbounds M[i, i] += lambda * Jd[i]
    end
    n2R = sum(abs2, R)
    Δx .= M \ JtR
    # predicted residual squared norm
    n2PR = sum(abs2, R - J * Δx)
    # actual residual and its squared norm
    x_buf = copy(x)
    assign_update_step!(x_buf, -1.0, Δx, sd)
    stackedtime_R!(R, x_buf, x_buf, sd)
    n2AR = sum(abs2, R)
    # quality of step
    qual = (n2AR - n2R) / (n2PR - n2R)
    # adjust lambda
    if qual > 0.75
        # good quality, extend trust region
        lambda = max(1e-16, lambda / nu)
    elseif qual < 1e-3
        # bad quality, shrink trust region
        lambda = min(1e16, lambda * nu)
    end
    lm_params[:] = [lambda, nu, qual]
    return nothing
end

function _sim_lm_first_step(
    x::AbstractArray{Float64},
    Δx::AbstractArray{Float64},
    sd::StackedTimeSolverData;
    lm_params=[0.1, 8.0, 0.0]
)

    lambda, nu, qual = lm_params

    R, FJ = stackedtime_RJ(x, x, sd; factorization=:none)
    J = FJ.A

    nx = length(R)
    JtR = J'R
    M = J'J
    # max is to make sure no zeros in Jd
    Jd = max.(1e-10, vec(sum(abs2, J; dims=1)))
    n2R = sum(abs2, R)

    qual = 0.0
    coef = 1.0
    n2AR = -1.0
    while qual < 1e-3
        for i = 1:nx
            @inbounds M[i, i] += lambda * Jd[i]
            # the above should be diag(M) += lambda * Jd
            # we have coef = (1-1/nu) for the iterations after the first one
            # in order to subtract the previous lambda from diag(M)
            # each iteration we shrink the trust region by setting lambda *= nu
        end
        Δx .= M \ JtR
        # predicted residual squared norm
        n2PR = sum(abs2, R - J * Δx)
        # actual residual and its squared norm
        x_buf = copy(x)
        assign_update_step!(x_buf, -1.0, Δx, sd)
        qual = 0.0
        try
            stackedtime_R!(R, x_buf, x_buf, sd)
            n2AR = sum(abs2, R)
            # quality of step
            qual = (n2AR - n2R) / (n2PR - n2R)
        catch
        end
        if qual > 0.75 # very good quality, extend the trust region
            lambda = lambda / nu
            break
        end
        if lambda >= 1e16 # lambda is getting too large
            lambda = 1e16
            break
        end
        # shrink the trust regions
        lambda = lambda * nu
        coef = 1.0 - 1.0 / nu
    end
    lm_params[:] = [lambda, nu, qual]
    return nothing
end

"""
    sim_lm!(x, sd, maxiter, tol, verbose, damping)

Solve the simulation problem using the Levenberg–Marquardt method.
  * `x` - the array of data for the simulation. All initial, final and exogenous
    conditions are already in place.
  * `sd::AbstractSolverData` - the solver data constructed for the simulation
    problem.
  * `maxiter` - maximum number of iterations.
  * `tol` - desired accuracy.
  * `verbose` - whether or not to print progress information.
  * `damping` - no damping is implemented here, however the damping is forwarded
    to [`sim_nr!`](@ref) when the LM algorithm switches to Newton-Raphson. 
"""
function sim_lm!(x::AbstractArray{Float64}, sd::StackedTimeSolverData,
    maxiter::Int64, tol::Float64, verbose::Bool,
    damping::Function # no damping in LM method, but passed on to sim_nr! when we switch to it
)

    # @warn "Levenberg–Marquardt method is experimental."
    if verbose
        @info "Simulation using Levenberg–Marquardt method."
    end

    lm_params = [0.1, 8.0, 0.0]
    Δx = Vector{Float64}(undef, size(sd.J, 1))
    R = Vector{Float64}(undef, size(sd.J, 1))
    stackedtime_R!(R, x, x, sd)
    nR0 = 1.0
    nR0 = norm(R, Inf)
    if verbose
        @info "0, || Fx || = $(nR0)"
    end
    # step 1
    _sim_lm_first_step(x, Δx, sd; lm_params)
    assign_update_step!(x, -1.0, Δx, sd)
    stackedtime_R!(R, x, x, sd)
    nR = norm(R, Inf)
    nΔx = norm(Δx, Inf)
    if verbose
        @info "1, || Fx || = $(nR), || Δx || = $(nΔx), λ = $(lm_params[1]), qual = $(lm_params[3])"
    end
    for it = 2:maxiter
        if nR < tol
            return true
        end
        _sim_lm_step(x, Δx, sd; lm_params)
        assign_update_step!(x, -1.0, Δx, sd)
        stackedtime_R!(R, x, x, sd)
        nR = norm(R, Inf)
        nΔx = norm(Δx, Inf)
        if verbose
            @info "$it, || Fx || = $(nR), || Δx || = $(nΔx), λ = $(lm_params[1]), qual = $(lm_params[3])"
        end
        if (nΔx < 1e-5 || true) && ((4nR < nR0 && nR < 1e-2) || (nR / nR0 > 0.8)) && (lm_params[1] <= 1e-4)
            if verbose
                @info "    --- switching to Newton-Raphson ---   "
            end
            sd.J_factorized[] = nothing
            return sim_nr!(x, sd, maxiter - it, tol, verbose, damping)
        end
        # nR0 = nR
    end
    return false
end

