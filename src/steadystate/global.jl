##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

_getparams(eqn) = hasproperty(eqn.eval_resid, :params) ? eqn.eval_resid.params :
                  hasproperty(eqn.eval_resid, :s) ? _getparams(eqn.eval_resid.s.eqn) :
                  ()
function inadmissible_error(eqind, eqn, point, val)
    vars = tuple(eqn.vsyms...)
    args = tuple((eval(Expr(:(=), sym, val)) for (sym, val) in zip(eqn.vsyms, point[eqn.vinds]))...)
    varsargs = (; (v => a for (v, a) in zip(vars, args))...)
    params = (; _getparams(eqn)...)
    if typeof(val) <: AbstractArray
        @error "Singular gradient in steady state equation $eqind: $eqn" params varsargs val
        error("Singular gradient in steady state equation $eqind: $eqn")
    else
        @error "Inadmissible point in steady state equation $eqind: $eqn" params varsargs val
        error("Inadmissible point in steady state equation $eqind: $eqn")
    end
end


"""
    global_SS_R!(R, point, equations)

Compute the residual vector of the given set of equations at the given point.
The `equations` argument can be a container whose `eltype` is
`SteadyStateEquations`. The residual `R` is updated in place.
"""
function global_SS_R!(resid::AbstractVector{Float64}, point::AbstractVector{Float64}, eqns::EqnIter) where {EqnIter}
    if !(eltype(eqns) <: SteadyStateEquation)
        error("Expected a set of steady state equations, not $(EqnIter).")
    end
    for (eqind, eqn) in enumerate(eqns)
        rr = try
            eqn.eval_resid(point[eqn.vinds])
        catch
            NaN64
        end
        if isnan(rr) || isinf(rr)
            inadmissible_error(eqind, eqn, point, rr)
            # vars = tuple(eqn.vsyms...)
            # args = tuple((eval(Expr(:(=), sym, val)) for (sym, val) in zip(eqn.vsyms, point[eqn.vinds]))...)
            # @error "Inadmissible point in equation $eqind: $eqn" vars args rr
            # error("Inadmissible point in equation $eqind: $eqn")
        end
        resid[eqind] = rr
    end
    return nothing
end
@assert precompile(global_SS_R!, (Vector{Float64}, Vector{Float64}, Vector{SteadyStateEquation}))

"""
    global_SS_R!(R, point, model::Model)

When a model is given, we compute the residual of the entire steady state
system.
"""
global_SS_R!(resid::AbstractVector{Float64}, point::AbstractVector{Float64}, model::Model) = global_SS_R!(resid, point, ModelBaseEcon.alleqns(model.sstate))
@assert precompile(global_SS_R!, (Vector{Float64}, Vector{Float64}, Model))

"""
    R, J = global_SS_RJ(point, equations)

Compute the residual vector `R` and the Jacobian matrix of the given set of
equations at the given point. The `equations` argument can be a container whose
`eltype` is `SteadyStateEquations`.
"""
function global_SS_RJ(point::AbstractVector{Float64}, eqns::EqnIter) where {EqnIter}
    if !(eltype(eqns) <: SteadyStateEquation)
        error("Expected a set of steady state equations, not $(EqnIter).")
    end
    neqns = length(eqns)
    nvars = length(point)
    R = zeros(neqns)
    J = zeros(neqns, nvars)
    for (eqind, eqn) in enumerate(eqns)
        rr, jj = try
            eqn.eval_RJ(point[eqn.vinds])
        catch
            NaN64, fill(NaN64, size(eqn.vinds))
        end
        if isnan(rr) || isinf(rr) || any(@. isnan(jj) | isinf(jj))
            inadmissible_error(eqind, eqn, point, rr)
            # args = tuple((eval(Expr(:(=), sym, val)) for (sym, val) in zip(eqn.vsyms, point[eqn.vinds]))...)
            # vars = tuple(eqn.vsyms...)
            # @error "Inadmissible point in equation $eqind: $eqn" vars args rr
            # error("Inadmissible point in equation $eqind: $eqn")
        end
        if any(@. isnan(jj) | isinf(jj))
            inadmissible_error(eqind, eqn, point, jj)
            #     args = tuple((eval(Expr(:(=), sym, val)) for (sym, val) in zip(eqn.vsyms, point[eqn.vinds]))...)
            #     vars = tuple(eqn.vsyms...)
            #     @error "Singular gradient in equation $eqind: $eqn" vars args rr
            #     error("Singular gradient in equation $eqind: $eqn")
        end
        R[eqind] = rr
        J[eqind, eqn.vinds] .= jj
    end
    return R, J
end
@assert precompile(global_SS_RJ, (Vector{Float64}, Vector{SteadyStateEquation}))

"""
    R, J = global_SS_RJ(point, model::Model)

When a model is given, we compute the residual of the entire steady state
system.
"""
global_SS_RJ(point::Vector{Float64}, model::Model) = global_SS_RJ(point, ModelBaseEcon.alleqns(model.sstate))
@assert precompile(global_SS_RJ, (Vector{Float64}, Model))
