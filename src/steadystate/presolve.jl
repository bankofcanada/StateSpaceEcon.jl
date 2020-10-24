##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

"""
    presolve_sstate!(model; <options>)
    presolve_sstate!(model, mask, values; <options>)

Solve for the steady state variables that are decoupled from the
system, or can be solved by forward substitution.

This is called automatically by the steady state solver before
running its main loop.

### Arguments
  * `model` - the Model instance
  * `mask` - a vector of Bool. Defaults to `model.sstate.mask`
  * `values` - a vector of numbers. Defaults to `model.sstate.values`
Caller must specify either both `mask` and `values` or neither of them.
`mask[i]` equals `true` if and only if the i-th steady state value has
alredy been solved for.
`mask` and `values` are both input and output data to the algorithm.
Any values that are successfully presolved are updated in place,
and their `mask` enties are set to `true`.

### Options
  * `verbose` - `true` or `false`, whether or not to print diagnostic messages.
  * `tol` - accuracy of the 1d solver.

"""
@inline presolve_sstate!(model::Model; kwargs...) = presolve_sstate!(model, model.sstate.mask, model.sstate.values; kwargs...)
function presolve_sstate!(model::Model, mask::AbstractVector{Bool}, values::AbstractVector{Float64};
    verbose = model.options.verbose, tol = model.options.tol,
    _1dsolver = :bisect)

    if _1dsolver == :newton
        solve1d = (sseqn, vals, ind)->newton1!(sseqn.eval_RJ, vals, ind; maxiter = 5, tol = tol)
    elseif _1dsolver == :bisect
        solve1d = (sseqn, vals, ind)->begin
            _, J = sseqn.eval_RJ(vals)
            bisect!(sseqn.eval_resid, vals, ind, J[ind]; maxiter = 1000, tol = tol)
        end
    else
        error("Unknown 1d solver $(_1dsolver)")
    end
    local alleqns = ModelBaseEcon.alleqns(model.sstate)
    while true
        # Keep looping for as long as we keep updating values
        updated = false
        for (eqind, sseqn) ∈ enumerate(alleqns)
            unsolved = .! mask[sseqn.vinds]
            if sum(unsolved) == 1
                # There is exactly one variable in this equation that's not solved yet.
                ind = findall(unsolved)[1]
                vals = view(values, sseqn.vinds)
                sym = ModelBaseEcon.ss_symbol(model.sstate, sseqn.vinds[ind])
                # store the current value, so we can restore it if 1d solver fails.
                foo = vals[ind]

                if solve1d(sseqn, vals, ind)
                    if abs(vals[ind]) < 1e-10
                        vals[ind] = 0.0
                    end
                    if verbose
                        pt_info = NamedTuple{tuple(sseqn.vsyms...)}(vals)
                        @info "Presolved $eqind: $sseqn for $sym = $(vals[ind])" pt_info
                    end
                    mask[sseqn.vinds[ind]] = true
                    updated = true
                else
                    vals[ind] = foo
                    if verbose
                        pt_info = NamedTuple{tuple(sseqn.vsyms...)}(vals)
                        @info "Failed to presolve $eqind: $sseqn for $sym" pt_info
                    end
                end
            end
        end
        #
        if ! updated
            break
        end
    end
    nothing
end
@assert precompile(presolve_sstate!, (Model, Vector{Bool}, Vector{Float64}))

