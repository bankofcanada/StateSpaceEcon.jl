##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

"""
    sim_gn!(x, sd, maxiter, tol, verbose [, linesearch])

Solve the simulation problem using the Gauss-Newton method.
  * `x` - the array of data for the simulation. All initial, final and exogenous
    conditions are already in place.
  * `sd::AbstractSolverData` - the solver data constructed for the simulation
    problem.
  * `maxiter` - maximum number of iterations.
  * `tol` - desired accuracy.
  * `verbose` - whether or not to print progress information.
"""
function sim_gn!(x::AbstractArray{Float64}, sd::StackedTimeSolverData,
    maxiter::Int64, tol::Float64, verbose::Bool, linesearch::Bool=false, kwargs...
)

    @warn "Gauss-Newton method is experimental."
    if verbose
        @info "Simulation using Gauss-Newton method."
    end

    Δx = Vector{Float64}(undef, size(sd.J, 1))
    R = Vector{Float64}(undef, size(sd.J, 1))
    stackedtime_R!(R, x, x, sd)
    nR0 = 1.0
    nR0 = norm(R, Inf)
    if verbose
        @info "0, || Fx || = $(nR0)"
    end
    nR = nR0
    Δx = similar(R)
    for it = 1:maxiter
        if nR < tol
            return true
        end
        R, FJ = stackedtime_RJ(x, x, sd; factorization=:none)
        nR = norm(R, Inf)
        J = FJ.A
        JtR = J'R
        JtJ = J'J
        Δx .= JtJ \ JtR
        nΔx = norm(Δx, Inf)
        assign_update_step!(x, -1.0, Δx, sd)
        if verbose
            @info "$it, || Fx || = $(nR), || Δx || = $(nΔx)"
        end
        if nΔx < tol
            return true
        end
    end
    return false
end

