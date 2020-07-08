

function inadmissible_error(eqind, eqn, point)
    vars = tuple(eqn.vsyms...)
    args = tuple((eval(Expr(:(=), sym, val)) for (sym, val) in zip(eqn.vsyms, point[eqn.vinds]))...)
    @error "Inadmissible point in equation $eqind: $eqn" vars args rr
    error("Inadmissible point in equation $eqind: $eqn")
end


"""
    global_SS_R!(res, point, equations)

Compute the residual vector of the given set of equations at the given point.
The `equations` argument can be a container or an iterator whose `eltype` is `SteadyStateEquations`.
The residual `res` is updated in place.
"""
function global_SS_R!(resid::AbstractVector{Float64}, point::AbstractVector{Float64}, eqns::EqnIter) where EqnIter
    if ! (eltype(eqns) <: SteadyStateEquation)
        error("Expected a set of steady state equations, not $(EqnIter).")
    end
    for (eqind, eqn) in enumerate(eqns)
        rr = try eqn.eval_resid(point[eqn.vinds]) catch; NaN64 end
        if isnan(rr) || isinf(rr)
            inadmissible_error(eqind, eqn, point)
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
    global_SS_R!(res, point, model::Model)

When a model is given, we compute the residual of the entire steady state system.
"""
@inline global_SS_R!(resid::AbstractVector{Float64}, point::AbstractVector{Float64}, model::Model) = global_SS_R!(resid, point, model.sstate.alleqns)
@assert precompile(global_SS_R!, (Vector{Float64}, Vector{Float64}, Model))




"""
    global_SS_RJ(point, equations)

Compute the residual vector and the Jacobian matrix of the given set of equations at the given point.
The return value is a tuple with two values.
The `equations` argument can be a container or an iterator whose `eltype` is `SteadyStateEquations`.
"""
function global_SS_RJ(point::AbstractVector{Float64}, eqns::EqnIter) where EqnIter
    if ! (eltype(eqns) <: SteadyStateEquation)
        error("Expected a set of steady state equations, not $(EqnIter).")
    end
    neqns = length(eqns)
    nvars = length(point)
    R = zeros(neqns)
    J = zeros(neqns, nvars)
    for (eqind, eqn) in enumerate(eqns)
        rr, jj = try eqn.eval_RJ(point[eqn.vinds]) catch; NaN64, fill(NaN64, size(eqn.vinds)) end
        if isnan(rr) || isinf(rr) || any(@. isnan(jj) | isinf(jj))
            inadmissible_error(eqind, eqn, point)
            # args = tuple((eval(Expr(:(=), sym, val)) for (sym, val) in zip(eqn.vsyms, point[eqn.vinds]))...)
            # vars = tuple(eqn.vsyms...)
            # @error "Inadmissible point in equation $eqind: $eqn" vars args rr
            # error("Inadmissible point in equation $eqind: $eqn")
        end
        # if any(@. isnan(jj) | isinf(jj))
        #     inadmissible_error()
        #     args = tuple((eval(Expr(:(=), sym, val)) for (sym, val) in zip(eqn.vsyms, point[eqn.vinds]))...)
        #     vars = tuple(eqn.vsyms...)
        #     @error "Singular gradient in equation $eqind: $eqn" vars args rr
        #     error("Singular gradient in equation $eqind: $eqn")
        # end
        R[eqind] = rr
        J[eqind,eqn.vinds] .= jj
    end
    return R, J
end
@assert precompile(global_SS_RJ, (Vector{Float64}, Vector{SteadyStateEquation}))

"""
    global_SS_RJ(point, model::Model)

When a model is given, we compute the residual of the entire steady state system.
"""
@inline global_SS_RJ(point::Vector{Float64}, model::Model) = global_SS_RJ(point, model.sstate.alleqns)
@assert precompile(global_SS_RJ, (Vector{Float64}, Model))
