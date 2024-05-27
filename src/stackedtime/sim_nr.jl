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
    - `damping_bank_rose(delta=0.1, rateK=10.0)` implements a the damping
      algorithm of Bank and Rose 1980
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
    while (it < maxiter) && (tol < nFx) && (λ*tol < nΔx)
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
function damping_armijo(; alpha::Real=1e-4, sigma::Real=0.5, lambda_min::Real=0.00001, verbose::Bool=false)
    α = convert(Float64, alpha)
    σ = convert(Float64, sigma)
    λ_min = convert(Float64, lambda_min)
    nF2_it = 0  # iteration number at which nF2 is valid
    nF2 = NaN   # the norm of the residual at the beginning of iteration nF2_it
    return function (it::Int, λ::Float64, nF::Float64, F::AbstractVector{Float64},
        ::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
        ::Union{Nothing,AbstractVector{Float64}}=nothing
    )
        # @printf "    it=%d, λ=%g, nF=%g\n" it λ nF
        it < 1 && return true, 1.0
        if nF2_it != it
            # first time we're called this iteration 
            nF2 = norm(F, 2)  # store the residual 
            nF2_it = it
            return false, 1.0   # try λ=1.0, a full Newton step, first
        end
        if λ < λ_min
            # λ too small
            verbose && @warn "Linesearch failed."
            return true, λ
        end
        if norm(F, 2) < (1.0 - α * λ) * nF2
            # Armijo test pass => accept the given λ
            return true, λ
        else
            # reject and try a smaller λ
            return false, σ * λ
        end
    end
end

# Bank, R.E., Rose, D.J. Global approximate Newton methods. Numer. Math. 37, 279–295 (1981). 
# https://doi.org/10.1007/BF01398257
function damping_br81(; delta::Real=0.1, rateK::Real=10.0, lambda_min::Real=1e-5, verbose::Bool=false)
    δ = convert(Float64, delta)
    λ_min = convert(Float64, lambda_min)
    bigK = 0.0  # Initialize with 0.0 (effectively the full Newton step)
    nF2_it = 0  # iteration number at which nF2 is valid
    nF2 = NaN   # the norm of the residual at the beginning of iteration nF2_it
    @inline calc_λ() = 1.0 / (1.0 + bigK * nF2)
    return function (it::Int, λ::Float64, nF::Float64, F::AbstractVector{Float64},
        ::Union{Nothing,Factorization,AbstractMatrix{Float64}}=nothing,
        ::Union{Nothing,AbstractVector{Float64}}=nothing
    )
        # @printf "    it=%d, λ=%g, nF=%g\n" it λ nF
        it < 1 && (bigK = 0.0; return true, 1.0)
        if nF2_it != it
            # first time we're called this iteration 
            nF2 = norm(F, 2)  # store the residual 
            nF2_it = it
            return false, calc_λ()
        end
        if (1 - δ * λ) * nF2 < norm(F, 2)
            # test failed => reject and try smaller λ
            if bigK == 0.0
                bigK = 1.0
            else
                bigK = rateK * bigK
            end
            λ = calc_λ()
            if λ > λ_min
                return false, λ
            else
                # λ too small
                verbose && @warn "Linesearch failed."
                return true, λ_min
            end
        else
            # lower bigK for next iteration ...
            bigK = bigK / rateK
            # ... and accept given λ 
            return true, λ
        end
    end
end
