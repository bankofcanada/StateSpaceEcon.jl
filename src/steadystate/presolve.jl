##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    presolve_sstate!(model; <options>)
    presolve_sstate!(model, mask, values; <options>)

Solve for the steady state variables that are decoupled from the system, or can
be solved by forward substitution.

This is called automatically by the steady state solver before running its main
loop.

### Arguments
  * `model` - the Model instance
  * `mask` - a vector of Bool. Defaults to `model.sstate.mask`
  * `values` - a vector of numbers. Defaults to `model.sstate.values` Caller
    must specify either both `mask` and `values` or neither of them. `mask[i]`
    equals `true` if and only if the i-th steady state value has alredy been
    solved for.

`mask` and `values` are both input and output data to the algorithm. Any values
that are successfully presolved are updated in place, and their `mask` enties
are set to `true`.

### Options
  * `verbose` - `true` or `false`, whether or not to print diagnostic messages.
  * `tol` - accuracy of the 1d solver.

"""
function presolve_sstate! end

function solve1d(sseqn, vals, ind, ::Val{:bisect}, tol = 1e-12, maxiter = 1000)
    R, J = sseqn.eval_RJ(vals)
    if abs(R) < tol && abs(J[ind]) > tol
        return true
    end
    return bisect!(sseqn.eval_resid, vals, ind, J[ind]; tol, maxiter)
end

function solve1d(sseqn, vals, ind, ::Val{:newton}, tol = 1e-12, maxiter = 5)
    return newton1!(sseqn.eval_RJ, vals, ind; tol, maxiter)
end

function _presolve_equations!(eqns, mask, values, method, verbose, tol)
    eqns_solved = falses(size(eqns))
    eqns_resid = zeros(size(eqns))
    retval = false # return value: true if any mask changed, false otherwise
    updated = true  # keep track if any mask changed within the loop
    while updated && !all(eqns_solved)
        updated = false # keep track if anything changes this outer iteration
        for (eqn_idx, sseqn) in enumerate(eqns)
            eqns_solved[eqn_idx] && continue
            # mask is true for variables that are already solved
            unsolved = .!mask[sseqn.vinds]
            nunsolved = sum(unsolved)
            if nunsolved == 0
                # all variables are solved, yet equation is not marked solved. 
                # check if equation is satisfied
                eqns_resid[eqn_idx] = R = sseqn.eval_resid(values[sseqn.vinds])
                # mark it solved either way, but issue a warning if residual is not zero
                eqns_solved[eqn_idx] = true
                if verbose && abs(R) > 100tol
                    @warn "Equation $eqn_idx has residual $R:\n    $sseqn"
                end
            elseif nunsolved == 1
                # only one variable left unsolved. call the 1d solver
                ind = findall(unsolved)[1]
                vals = values[sseqn.vinds]
                success = solve1d(sseqn, vals, ind, Val(method), 0.1tol)
                if success
                    if abs(vals[ind]) < 1e-10
                        vals[ind] = 0.0
                    end
                    values[sseqn.vinds[ind]] = vals[ind]
                    retval = updated = mask[sseqn.vinds[ind]] = true
                    if verbose
                        @info "Presolved equation $eqn_idx for $(sseqn.vsyms[ind]) = $(vals[ind])\n    $sseqn"
                    end
                else
                    if verbose
                        @info "Failed to presolve $eqn_idx for $(sseqn.vsyms[ind])\n    $sseqn"
                    end
                end
            else
                nunsolved > 1
                continue
            end
        end
    end
    return retval
end



@inline presolve_sstate!(model::Model; kwargs...) =
    presolve_sstate!(model, model.sstate.mask, model.sstate.values; kwargs...)
@inline presolve_sstate!(model::Model, mask::AbstractVector{Bool}, values::AbstractVector{Float64};
    verbose = model.verbose, tol = model.tol, _1dsolver = :bisect, method = _1dsolver) =
    _presolve_equations!(ModelBaseEcon.alleqns(model.sstate), mask, values, method, verbose, tol)
@inline presolve_sstate!(eqns::Vector{SteadyStateEquation}, mask::AbstractVector{Bool}, values::AbstractVector{Float64};
    verbose = false, tol = 1e-12, _1dsolver = :bisect, method = _1dsolver) =
    _presolve_equations!(eqns, mask, values, method, verbose, tol)

@assert precompile(presolve_sstate!, (Model, Vector{Bool}, Vector{Float64}))
@assert precompile(presolve_sstate!, (Vector{SteadyStateEquation}, Vector{Bool}, Vector{Float64}))

