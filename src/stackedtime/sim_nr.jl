##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

"""
    sim_nr!(x, sd, maxiter, tol, verbose, damping)

Solve the simulation problem using a Newton iteration with damping.
  * `x` - the array of data for the simulation. All initial, final and exogenous
    conditions are already in place.
  * `sd::AbstractSolverData` - the solver data constructed for the simulation
    problem.
  * `maxiter::Int` - maximum number of iterations.
  * `tol::Float64` - desired accuracy.
  * `verbose::Bool` - whether or not to print progress information.
  * `damping::Function` - a callback function that implements the damping logic.
    Built-in versions include
    - `damping_none` always returns λ=1.0, no damping, resulting in the standard
      Newton-Raphson method
    - `damping_schedule(value)` always returns `λ=value`
    - `damping_schedule(vector)` returns `λ=vector[it]` on iteration `it`. If
      the vector is shorter than the number of iterations, it keeps returning
      `array[end]`.
    - `damping_amijo(α=1e-4, σ=0.5)` implements a standard linesearch algorithm
      based on the Armijo rule
    - `damping_br81(delta=0.1, rateK=10.0)` implements a the damping
      algorithm of Bank and Rose 1981

##### Conventions for custom damping function.
The `damping` callback function is expected to have the following signature:

    function custom_damping(k::Int, λ::Float64, nR::Float64, R::AbstractVector{Float64},
        J::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
        Δx::Union{Nothing,AbstractVector{Float64}}=nothing
    )::Tuple{Bool, Float64}
        # <your code goes here>
    end

The first call will be with `k=0`, before the solver enters the Newton
iterations loop. This should allow any initialization and defaults to be setup.
In this call, the values of `R` and `nR` will equal the residual and its norm at
the initial guess. The norm used is `nR = norm(R, Inf)`. The `J` and `Δx` are
not available. 

Each subsequent call will be with `k` between 1 and `maxiter` (possibly multiple
calls with the same `k`) will have the current `λ` (which equals the one returned by the
previous call), the current `R` (and its Inf-norm `nR`), the Jacobian `J` and
the Newton direction `Δx`.

The damping function must return a tuple `(accept, λ)`. The same Newton
iteration `k` will continue until the damping function returns `accept=true`,
after which will begin the next Newton iteration (`k=k+1``).

All calls with the same `k` will have the same `J` and `Δx` but different `R`
and `nR`. These equal the Jacobian (possibly factorized), computed at the
iteration guess, `xₖ` (not provided), and the Newton direction for the
iteration.

The first time the damping function is called for a given `k`, the `R` equals
the residual at `xₖ` and the Newton direction is `Δx = J \\ R` for the `R` at
this time.

Each subsequent call with the same `k` will have the residual, `R`, (and its
norm) computed at `xₖ - λ Δx` (not provided). Your callback must decide whether
to accept this step, by returning `(true, λ)`, or reject it and propose a new λ
to try, by returning `(false, new_λ)`. Don't return `(false, λ)` because this
will make it an infinite loop. Good luck!

"""
function sim_nr!(x::AbstractArray{Float64}, sd::StackedTimeSolverData,
    maxiter::Int64, tol::Float64, verbose::Bool, damping::Function
)
    Fx = Vector{Float64}(undef, size(sd.J, 1))
    Fx = stackedtime_R!(Fx, x, x, sd)
    xtmp = copy(x)
    it = 0
    nFx = norm(Fx, Inf)
    nΔx = Inf
    _, λ = damping(0, 1.0, nFx, Fx)::Tuple{Bool,Float64}
    while (it <= maxiter) && (tol < nFx) && (λ*tol < nΔx)
        it = it + 1
        TMP, Jx = stackedtime_RJ(x, x, sd)
        copyto!(Fx, TMP)
        nFx = norm(Fx, Inf)
        Δx = sf_solve!(Jx, TMP)  #! overwrites TMP
        # search for λ
        accepted, λ = damping(it, λ, nFx, Fx, Jx, Δx)::Tuple{Bool,Float64}
        while !accepted
            copyto!(xtmp, x)
            assign_update_step!(xtmp, -λ, Δx, sd)
            nFtmp = try
                stackedtime_R!(Fx, xtmp, xtmp, sd)
                norm(Fx, Inf)
            catch
                NaN
            end
            accepted, λ = damping(it, λ, nFtmp, Fx, Jx, Δx)::Tuple{Bool,Float64}
        end
        assign_update_step!(x, -λ, Δx, sd)
        nΔx = λ * norm(Δx, Inf)
        verbose && @info "$it, || Fx || = $(nFx), || Δx || = $(nΔx), λ = $λ"
        Fx = stackedtime_R!(Fx, x, x, sd)
        nFx = norm(Fx, Inf)
    end
    verbose && @info "$it, || Fx || = $(nFx)"
    return it <= maxiter
end



function damping_none(it::Int, λ::Float64, nR::Float64, R::AbstractVector{Float64},
    J::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
    Δx::Union{Nothing,AbstractVector{Float64}}=nothing
)
    return true, 1.0
end

function damping_schedule(lambda::Real; verbose::Bool=false)
    λ = convert(Float64, lambda)::Float64
    return function (::Int, ::Float64, ::Float64, ::AbstractVector{Float64},
        ::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
        ::Union{AbstractVector{Float64},Nothing}=nothing
    )
        return true, λ
    end
end

function damping_schedule(lambda::AbstractVector{<:Real}; verbose::Bool=false)
    λ = Float64[lambda...]
    return function (it::Int, ::Float64, ::Float64, ::AbstractVector{Float64},
        ::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
        ::Union{AbstractVector{Float64},Nothing}=nothing
    )
        it < 1 && return true, 1.0
        index = min(length(λ), it)
        return @inbounds true, λ[index]
    end
end

# the Armijo rule: C.T.Kelly, Iterative Methods for Linear and Nonlinear Equations, ch.8.1, p.137
function damping_armijo(; alpha::Real=1e-4, sigma::Real=0.5, lambda_min::Real=1e-5, lambda_max::Real=1.0, lambda_growth::Real=1.1, verbose::Bool=false)
    α = convert(Float64, alpha)
    σ = convert(Float64, sigma)
    λ_min = convert(Float64, lambda_min)
    λ_max = convert(Float64, lambda_max)
    λ_growth = convert(Float64, lambda_growth)
    nF2_it = 0  # iteration number at which nF2 is valid
    nF2 = NaN   # the norm of the residual at the beginning of iteration nF2_it
    return function (it::Int, λ::Float64, nF::Float64, F::AbstractVector{Float64},
        ::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
        ::Union{Nothing,AbstractVector{Float64}}=nothing
    )
        it < 1 && return true, min(1.0, λ_max)  

        if nF2_it != it
            # First call in this iteration: Store the residual norm
            nF2 = norm(F, 2)
            nF2_it = it
            return false, min(1.0, λ_max)  
        end

        if λ < λ_min
            verbose && @warn "Linesearch failed: λ fell below λ_min."
            return true, λ
        end

        if norm(F, 2) < (1.0 - α * λ) * nF2
            # Armijo test passed => accept the given λ
            new_λ = min(λ * λ_growth, λ_max)  # Gradually increase λ but cap at λ_max
            
            if abs(norm(F, 2) - nF2) < 1e-12  # Convergence check to break loops
                verbose && @info "Solver converged: residual change too small."
                return true, new_λ
            end

            return true, new_λ
        else
            # Reject and try a smaller λ
            new_λ = max(σ * λ, λ_min)
            
            if λ == new_λ  # Prevent infinite shrinking loops
                verbose && @warn "Stuck in shrinking loop, forcing exit."
                return true, λ
            end

            return false, new_λ
        end
    end
end

# Bank, R.E., Rose, D.J. Global approximate Newton methods. Numer. Math. 37, 279–295 (1981). 
# https://doi.org/10.1007/BF01398257
function damping_br81(; delta::Real=0.1, rateK::Real=10.0, lambda_min::Real=1e-5, lambda_max::Real=1.0, lambda_growth::Real=1.05, verbose::Bool=false)
    δ = convert(Float64, delta)
    λ_min = convert(Float64, lambda_min)
    λ_max = convert(Float64, lambda_max)
    λ_growth = convert(Float64, lambda_growth)
    bigK = 0.0  # Initialize with 0.0 (effectively the full Newton step)
    nF2_it = 0  # iteration number at which nF2 is valid
    nF2 = NaN   # the norm of the residual at the beginning of iteration nF2_it
    @inline calc_λ() = 1.0 / (1.0 + bigK * nF2)
    return function (it::Int, λ::Float64, nF::Float64, F::AbstractVector{Float64},
        ::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
        ::Union{Nothing,AbstractVector{Float64}}=nothing
    )
        # Initialization step
        it < 1 && (bigK = 0.0; return true, λ_max)
        
        if nF2_it != it
            # First time we're called in this iteration
            nF2 = norm(F, 2)  # store the residual 
            nF2_it = it
            return false, calc_λ()
        end
        
        if (1 - δ * λ) * nF2 < norm(F, 2)
            # If test failed, decrease step size
            if bigK == 0.0
                bigK = 1.0
            else
                bigK *= rateK  # Increase `bigK` slower to prevent excessive reductions in λ
            end
            λ = calc_λ()
            if λ > λ_min
                return false, λ
            else
                verbose && @warn "Linesearch failed."
                return true, λ_min
            end
        else
            # Lower `bigK` more aggressively when convergence is happening
            bigK /= sqrt(rateK)

            # If λ is near the lower bound for many steps, slowly increase it
            λ = min(λ * λ_growth, λ_max)

            return true, λ
        end
    end
end
