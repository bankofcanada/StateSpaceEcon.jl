##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
Newton-Raphson step of the steady state solver.
"""

# This line intentionally left blank.

"""
    NRData

A data structure that holds the necessary buffers and internal data for
performing a step of the Newton-Raphson algorithm

Internal use. Do not call directly.

"""
struct NRData
    sd::SolverData
    ss::SteadyStateData
    v_buffer::Vector{Float64}
    r_buffer::Vector{Float64}
    updated::BitArray{1}
    linesearch::Bool
end

NRData(model::Model; kwargs...) = NRData(model, SolverData(model); kwargs...)
NRData(model::Model, sd::SolverData; linesearch::Bool = false) = NRData(sd, model.sstate,
                    Array{Float64}(undef, sd.nvars), Array{Float64}(undef, sd.neqns), 
                    trues(sd.nvars), linesearch)

"""
    step_nr!(x, dx, resid, J, nr::NRData; verbose=false)

Attempt a Newton-Raphson step. The `nr` structure and the `dx` vector would be
updated accordingly. Vectors `x`, `resid` and the matrix `J` are read-only
inputs.

!!! warning
    Internal function, do not call directly.
"""
function step_nr!(xx::AbstractArray{Float64,1}, dx::AbstractArray{Float64,1},
                    rr::AbstractArray{Float64,1}, jj::AbstractArray{Float64,2},
                    nr::NRData; verbose::Bool = false)
    ff = qr(jj, Val(true))
    rj = rank(ff.R)
    # nr.r_buffer .= ff.Q' * rr
    # dx[ff.p[1:rj]] = ff.R[1:rj,1:rj] \ nr.r_buffer[1:rj]
    nr.updated[ff.p[1:rj]] .= true
    nr.updated[ff.p[rj + 1:end]] .= false
    dx .= ff \ rr
    if verbose && any(.!nr.updated)
        problem_vars = join((ModelBaseEcon.ss_symbol(nr.ss, vi) for vi in findall(nr.sd.solve_var)[.!nr.updated]), ", ")
        @warn "Unable to update $(problem_vars)."
    end
    nf = norm(rr)
    if nr.linesearch
        # the Armijo rule: C.T.Kelly, Iterative Methods for Linear and Nonlinear Equations, ch.8.1, p.137
        λ = 1.0
        α = 1e-4
        σ = 0.5
        while λ > 0.00001
            nr.v_buffer .= xx - λ .* dx
            nrb2 = 0.0
            try
                global_SS_R!(nr.r_buffer, nr.v_buffer, nr.sd)
                nrb2 = norm(nr.r_buffer)
            catch e
                nrb2 = Inf
            end
            if nrb2 < (1.0 - α * λ) * nf
                if verbose && λ < 1.0
                    @info "Linesearch success with λ = $λ."
                end
                dx .*= λ
                break
            end
            λ = σ * λ
        end
        if verbose && λ <= 0.00001
            @warn "Linesearch failed."
        end
    end
    return nothing
end
@assert precompile(step_nr!, (Array{Float64,1}, Array{Float64,1}, Array{Float64,1}, Array{Float64,2}, NRData))
