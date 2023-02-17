##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

using ModelBaseEcon.OrderedCollections

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
  * `vals` - a vector of numbers. Defaults to `model.sstate.values` Caller
    must specify either both `mask` and `vals` or neither of them. `mask[i]`
    equals `true` if and only if the i-th steady state value has alredy been
    solved for.

`mask` and `vals` are both input and output data to the algorithm. Any vals
that are successfully presolved are updated in place, and their `mask` enties
are set to `true`.

### Options
  * `verbose` - `true` or `false`, whether or not to print diagnostic messages.
  * `tol` - accuracy of the 1d solver.

"""
function presolve_sstate! end

function solve1d(sseqn, vals, ind, ::Val{:bisect}, tol = 1e-12, maxiter = 1000)
    R, J = try
        sseqn.eval_RJ(vals)
    catch
        return false
    end
    if abs(R) < tol && abs(J[ind]) > tol
        return true
    end
    return bisect!(sseqn.eval_resid, vals, ind, J[ind]; tol, maxiter)
end

function solve1d(sseqn, vals, ind, ::Val{:newton}, tol = 1e-12, maxiter = 5)
    return newton1!(sseqn.eval_RJ, vals, ind; tol, maxiter)
end

function _presolve_equations!(eqns, mask, vals, method, verbose, tol)
    eqns_solved = OrderedDict{Symbol, Bool}(key => false for key in keys(eqns))
    eqns_resid = OrderedDict{Symbol, Float64}(key => 0.0 for key in keys(eqns))
    # eqns_resid = zeros(length(eqns))
    retval = false # return value: true if any mask changed, false otherwise
    updated = true  # keep track if any mask changed within the loop
    while updated && !all(values(eqns_solved))
        updated = false # keep track if anything changes this outer iteration
        for (eqn_key, sseqn) in eqns
            eqns_solved[eqn_key] && continue
            # mask is true for variables that are already solved
            unsolved = .!mask[sseqn.vinds]
            nunsolved = sum(unsolved)
            if nunsolved == 0
                # all variables are solved, yet equation is not marked solved. 
                # check if equation is satisfied
                eqns_resid[eqn_key] = R = sseqn.eval_resid(vals[sseqn.vinds])
                # mark it solved either way, but issue a warning if residual is not zero
                eqns_solved[eqn_key] = true
                if verbose && abs(R) > 100tol
                    @warn "Equation $eqn_key has residual $R:\n    $sseqn"
                end
            elseif nunsolved == 1
                # only one variable left unsolved. call the 1d solver
                ind = findall(unsolved)[1]
                _vals = vals[sseqn.vinds]
                success = solve1d(sseqn, _vals, ind, Val(method), 0.1tol)
                if success
                    if abs(_vals[ind]) < 1e-10
                        _vals[ind] = 0.0
                    end
                    vals[sseqn.vinds[ind]] = _vals[ind]
                    retval = updated = mask[sseqn.vinds[ind]] = true
                    if verbose
                        @info "Presolved equation $eqn_key for $(sseqn.vsyms[ind]) = $(_vals[ind])\n    $sseqn"
                    end
                else
                    if verbose
                        @info "Failed to presolve $eqn_key for $(sseqn.vsyms[ind])\n    $sseqn"
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



presolve_sstate!(model::Model; kwargs...) =
    presolve_sstate!(model, model.sstate.mask, model.sstate.values; kwargs...)
presolve_sstate!(model::Model, mask::AbstractVector{Bool}, values::AbstractVector{Float64};
    verbose = model.verbose, tol = model.tol, _1dsolver = :bisect, method = _1dsolver) =
    _presolve_equations!(ModelBaseEcon.alleqns(model.sstate), mask, values, method, verbose, tol)
presolve_sstate!(eqns::LittleDict{Symbol,SteadyStateEquation}, mask::AbstractVector{Bool}, values::AbstractVector{Float64};
    verbose = false, tol = 1e-12, _1dsolver = :bisect, method = _1dsolver) =
    _presolve_equations!(eqns, mask, values, method, verbose, tol)

@assert precompile(presolve_sstate!, (Model, Vector{Bool}, Vector{Float64}))
# @assert precompile(presolve_sstate!, (LittleDict{Symbol,SteadyStateEquation}, Vector{Bool}, Vector{Float64}))

