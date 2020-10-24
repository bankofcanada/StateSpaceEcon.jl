##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

export check_sstate

"""
    check_sstate(model; <options>)

Run a diagnostic test to determine if the steady state solution stored within
the given model object is indeed a solution of the steady state system of
equations.

Return the number of steady state equations that are violated by the current
steady state solution. If `verbose=true`, also display diagnostic information in
the form of listing all bad equations and their residuals.

### Options
Standard options (default values from `model.options`)
  * `verbose` 
  * `tol` - an equation is considered satisfied if its residual, in absolute
    value, is smaller than this number.

"""
function check_sstate(model::Model; verbose::Bool = model.options.verbose, tol::Float64 = model.options.tol)
    R, J = global_SS_RJ(model.sstate.values, model)
    bad_eqn = 0
    bad_eqn_str = ""
    for (ind, res) ∈ sort(collect(enumerate(R)), lt = (a, b)->abs(a[2]) > abs(b[2]))
        if abs(res) > tol
            bad_eqn += 1
            if verbose
                eqn_str = ModelBaseEcon.geteqn(ind, model.sstate)
                res_str = @sprintf "%- 10g" res
                bad_eqn_str *= "\nE$ind  res=$(res_str)  $(eqn_str)"
            end
        end
    end
    if bad_eqn > 0 && verbose
        s = bad_eqn > 1 ? "equations are" : "equation is"
        b = bad_eqn > 1 ? "$(bad_eqn) " : ""
        @warn "The following $(b)steady state $(s) not satisfied: $(bad_eqn_str)"
    end
    if bad_eqn == 0 && verbose
        @info "All steady state equations are satisfied."
    end
    return bad_eqn
end
@assert precompile(check_sstate, (Model,))



export diagnose_sstate
"""
    diagnose_sstate([point,] model)

Run diagnostics on the steady state of the given model. If `point` is not given,
then we check the steady state solution stored inside the given model.

Return a tuple of "bad" equations and "bad" variables. 

The set of "bad" equations is one that is inconsistent, i.e. there is no
solution. This might happen if the system is overdetermined.

The set of "bad" variables contains variables that cannot be solved uniquely.
This might happen if the system is underdetermined. In this case, try adding
steady state constraints until you get a unique solution. See
[`@steadystate`](@ref ModelBaseEcon.@steadystate) in ModelBaseEcon.

!!! warning "Internal function"
    The output from this function may be difficult to read.<br>
    Call [`check_sstate`](@ref) instead.
"""
@inline diagnose_sstate(model::Model) = diagnose_sstate(model.sstate.values, model)
function diagnose_sstate(point::AbstractVector{Float64}, model::Model)
    tol = model.options.tol 
    rr, jj = global_SS_RJ(point, model)
    neqns, nvars = size(jj)
    ff = qr([jj Matrix(I, neqns, neqns); zeros(neqns, nvars) Matrix(I, neqns, neqns)], Val(true))
    sol = Vector{Float64}(undef, neqns + nvars)
    rj = rank(ff.R)
    rhs = ff.Q' * vcat(rr, zeros(neqns))
    sol[ff.p[1:rj]] = ff.R[1:rj,1:rj] \ rhs[1:rj]
    bad = falses(neqns)
    bad_res = abs.(sol[nvars + 1:end])
    bad[bad_res .> tol] .= true
    bad = sort(findall(bad), lt = (a, b)->bad_res[a] > bad_res[b])
    bad_eqn = tuple(("E$(be)  residual=$(bad_res[be])  $(geteqn(be, model.sstate))" for be in bad)...)
    vsyms = Symbol[ModelBaseEcon.ss_symbol(model.sstate, vi) for vi = 1:nvars]
    bad_var_mask = falses(size(vsyms))
    bad_var_mask[ff.p[rj + 1:end]] .= true
    if model.flags.ssZeroSlope
        bad_var_mask[2:2:end] .= false
    end
    for (vi, v) ∈ enumerate(model.allvars)
        if issteady(v)
            bad_var_mask[2vi] = false
        end
        if isshock(v)
            bad_var_mask[2vi-1] = false
            bad_var_mask[2vi] = false
        end
    end
    # bad_var = vsyms[filter(i->i <= nvars, findall(bad_var_mask))]
    bad_var = vsyms[bad_var_mask]
    return bad_eqn, bad_var
end
@assert precompile(diagnose_sstate, (Vector{Float64}, Model))
