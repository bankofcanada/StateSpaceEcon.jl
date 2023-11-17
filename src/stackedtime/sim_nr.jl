##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

"""
    sim_nr!(x, sd, maxiter, tol, verbose [, linesearch])

Solve the simulation problem.
  * `x` - the array of data for the simulation. All initial, final and exogenous
    conditions are already in place.
  * `sd::AbstractSolverData` - the solver data constructed for the simulation
    problem.
  * `maxiter` - maximum number of iterations.
  * `tol` - desired accuracy.
  * `verbose` - whether or not to print progress information.
  * `linesearch::Bool` - When `true` the Newton-Raphson is modified to include a 
    search along the descent direction for a sufficient decrease in f. It will 
    do this at each iteration. Default is `false`.

"""
@timeit_debug timer function sim_nr!(x::AbstractArray{Float64}, sd::StackedTimeSolverData,
    maxiter::Int64, tol::Float64, verbose::Bool, linesearch::Bool=false)
    for it = 1:maxiter
        Fx = stackedtime_R!(Vector{Float64}(undef, size(sd.J, 1)), x, x, sd)
        nFx = norm(Fx, Inf)
        if nFx < tol
            if verbose
                @info "$it, || Fx || = $(nFx)"
            end
            return true
        end
        Δx, Jx = stackedtime_RJ(x, x, sd)
        Δx = sf_solve!(Jx, Δx)

        λ = 1.0
        if linesearch
            nf = norm(Fx)
            # the Armijo rule: C.T.Kelly, Iterative Methods for Linear and Nonlinear Equations, ch.8.1, p.137
            α = 1e-4
            σ = 0.5
            while λ > 0.00001
                x_buf = copy(x)
                assign_update_step!(x_buf, -λ, Δx, sd)
                nrb2 = try
                    stackedtime_R!(Fx, x_buf, x_buf, sd)
                    norm(Fx)
                catch e
                    Inf
                end
                if nrb2 < (1.0 - α * λ) * nf
                    # if verbose && λ < 1.0
                    #     @info "Linesearch success with λ = $λ."
                    # end
                    break
                end
                λ = σ * λ
            end
            if verbose
                if λ <= 0.00001
                    @warn "Linesearch failed."
                elseif λ < 1.0
                    @info "Linesearch success with λ=$λ"
                end
            end
        end
        nΔx = λ * norm(vec(Δx), Inf)
        assign_update_step!(x, -λ, Δx, sd)
        if verbose
            @info "$it, || Fx || = $(nFx), || Δx || = $(nΔx)"
        end
        if nΔx < tol
            return true
        end
    end
    return false
end

