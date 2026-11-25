##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

export sssolve!
"""
    sssolve!(model; <options>)

Solve the steady state problem for the given model.

### Options
Standard options (default values are taken from `model.options`)
  * `verbose`
  * `tol`, `maxiter` - control the stopping criteria of the solver

Specific options
  * `presolve::Bool` - whether or not to use a presolve pass. Default is `true`.
  * `method::Symbol` - choose the solution algorithm. Valid options are `:nr`
    for Newton-Raphson, `:lm` for Levenberg-Marquardt, and `:auto`. The `:auto`
    method starts with the LM algorithm and automatically switches to NR when it
    starts to converge. Default is `:nr`.

### Exceptions from the model's equations
Only a `DomainError` thrown while evaluating the equations (for example `log`
or `sqrt` of a negative number) marks a point as inadmissible. Any other
exception from the equations propagates unchanged.

`sssolve!` needs a starting point (the values in `model.sstate`) at which every
equation can be evaluated; otherwise it stops with an "Inadmissible point"
`ErrorException`. After that it backs off from an inadmissible point only at
trial points: the Newton line search (enabled with `nropts = Options(linesearch
= true)`) and the first Levenberg-Marquardt step reject it and try a shorter
step, and an update of the auxiliary variables that hits one is skipped.
Anywhere else it stops with an "Inadmissible point" `ErrorException` (or
"Singular gradient" when only a derivative is not finite). Two limits: when the
line search rejects every step, the full Newton step is taken, and the point
reached by the last iteration is returned without being evaluated, so check the
result with [`check_sstate`](@ref).

The presolve, run by [`clear_sstate!`](@ref) and [`initial_sstate!`](@ref),
leaves an equation unsolved when it cannot be evaluated at the starting point.
With `model.options.substitutions = true`, `log(expr)` in an equation is
replaced by an auxiliary variable, so a negative `expr` does not throw a
`DomainError`.

If a function used in your equations cannot handle some arguments and you want
the solver to back off, throw a `DomainError` for them, for example
`f(x::Real) = x > 0 ? log(x) : throw(DomainError(x))`. Accept `Real`, not only
`Float64`: derivatives are computed with ForwardDiff dual numbers. Do not return
`NaN` instead, and do not turn every exception into a `DomainError`: that would
hide errors in the model.
"""
sssolve!(model::Model; kwargs...) = Base.invokelatest(_sssolve!, model; kwargs...)::Vector{Float64}
function _sssolve!(model::Model;
    verbose::Bool = model.options.verbose,
    tol::Float64 = model.options.tol,
    maxiter::Int64 = model.options.maxiter,
    presolve::Bool = true,
    nropts = Options(linesearch = false),
    lmopts = Options(),
    method::Symbol = :nr
)::Vector{Float64}

    if method ∉ (:nr, :lm, :auto)
        error("Method should be one of :nr, :lm, or :auto, not :$method")
    end
    
    ss = model.sstate
    ss_nvars = 2 * length(ss.vars)
    ss_neqns = ModelBaseEcon.neqns(ss)

    # !! make sure constraints use the latest parameter values
    # !! before calling SolverData constructor, because it computes the initial residual
    for eqn in values(ss.constraints)
        ModelBaseEcon._update_eqn_params!(eqn.eval_resid, model.parameters)
    end

    sd = SolverData(model; presolve = presolve)
    if sd.nvars == 0
        # Nothing left to solve for
        return sd.point
    end
    
    # nvars = length(model.sstate.mask)
    # vsyms = Symbol[ModelBaseEcon.ss_symbol(model.sstate, vi) for vi = 1:nvars]
    # presolved_var = vsyms[model.sstate.mask]

    lm = LMData(model, sd; lmopts...)
    nr = NRData(model, sd; nropts...)

    if verbose
        if sd.neqns < ss_neqns
            @info "Presolved $(ss_neqns - sd.neqns) equations for $(ss_nvars - sd.nvars) variables."
        end
        @info "Solving $(sd.neqns) equations for $(sd.nvars) variables."
    end

    ssvals = copy(ss.values)
    xx = ssvals[sd.solve_var]
    dx = similar(xx)

    undef_vars = Set{Int64}()

    if method ∈ (:lm, :auto)
        r0, j0 = global_SS_RJ(xx, sd)
        first_step_lm!(xx, dx, r0, j0, lm; verbose = verbose)
        nf = nr0 = norm(r0, Inf)
        if verbose
            @info "0, || Fx || = $(nr0), || Δx || = $(norm(dx, Inf)), lambda = $(lm.params[1])"
        end
        xx .-= dx
    else
        nr0 = 1.0
    end

    run_nr = (method == :nr)
    for it = 1:maxiter

        if any(isnan.(xx))
            error("Nan detected.")
        end

        r0, j0 = global_SS_RJ(xx, sd)
        nf = norm(r0, Inf)

        if (method == :nr) || ((method == :auto) && run_nr)
            step_nr!(xx, dx, r0, j0, nr; verbose = verbose)
            last_step_method = :nr
        else
            step_lm!(xx, dx, r0, j0, lm; verbose = verbose)
            run_nr = (nf < nr0 < 1e-2) && (lm.params[1] <= 1e-3)
            last_step_method = :lm
        end
        nr0 = nf
        xx .-= dx

        # try updating the auxiliary variables
        # (skipped if the residual is not finite before or after the update)
        if any(sd.solve_var[length(model.variables)*2+1:end]) && _active_SS_R!(r0, xx, sd) == 0
            n1 = sum(abs2, r0)
            ssvals[sd.solve_var] .= xx
            auxvals = try
                update_auxvars_ss(ssvals, model)
            catch err
                err isa DomainError || rethrow()
                nothing
            end
            if !isnothing(auxvals)
                ssvals .= auxvals
                if _active_SS_R!(r0, ssvals[sd.solve_var], sd) == 0 && sum(abs2, r0) < n1
                    xx .= ssvals[sd.solve_var]
                    if verbose
                        @info "    -- updated auxiliary variables"
                    end
                end
            end
        end

        nx = norm(dx, Inf)
        if verbose
            if last_step_method == :nr
                @info "$it, NR, || Fx || = $(nf), || Δx || = $(nx)"
            else
                @info "$it, LM, || Fx || = $(nf), || Δx || = $(nx), lambda = $(lm.params[1])"
            end
        end
        if nx < tol || nf < tol
            break
        end
    end
    xx[abs.(xx).<tol] .= 0.0
    ss.values[sd.solve_var] .= xx
    ss.mask[sd.solve_var] .= true
    if run_nr
        foo = (1:ss_nvars)[sd.solve_var] # global indexes of active unknowns
        for v in findall(.!nr.updated)
            # mark unknowns that were not updated as not solved
            ss.mask[foo[v]] = false
        end
    end
    if verbose && nf > tol
        bad_eqn, bad_var = diagnose_sstate(model)
        # bad_var = setdiff(bad_var,presolved_var)
        # lbva = length(bad_var)
        # if lbva > 0
        #     bad_var = join(bad_var, ",")
        #     vars_str = "Unable to solve for $(lbva) variables:\n   $bad_var"
        #     @warn vars_str
        # end
        lbeq = length(bad_eqn)
        if lbeq > 0
            bad_eqn = (lbeq > 10) ? (join(bad_eqn[1:10], "\n   ") * "\n   . . .") : join(bad_eqn, "\n   ")
            eqns_str = "System may be inconsistent. Couldn't solve $(lbeq) equations to the required accuracy:\n   $bad_eqn"
            @warn eqns_str
        end
    end
    return ss.values
end
precompile(_sssolve!, (Model,))