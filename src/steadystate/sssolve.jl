
export sssolve!
"""
    sssolve!(model; <options>)

Solve the steady state problem for the given model.

### Options
  * `verbose - defaults to `model.options.verbose`
  * `tol` - defaults to `model.options.tol`
  * `maxiter` - defaults to `model.options.maxiter`
  * `presolve::Bool` - whether or not the presolve the system.
  * `method::Symbol` - choose the solution algorithm. Valid options are `:nr`, `lm`, and `:auto`.
  The `:auto` method starts with the LM algorithm and automatically switches to NR when it
  starts to converge.
  * `nropts::Options` - options to control the Newton-Raphson algorithm.
  * `lmopts::Options` - options to control the Levenberg–Marquardt algorithm.
"""
function sssolve!(model::Model; 
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

    sd = SolverData(model; presolve = presolve)
    if sd.nvars == 0
        # Nothing left to solve for
        return sd.point
    end

    ss = model.sstate
    vars = ss.vars[sd.solve_var]
    ss_nvars = length(ss.vars)
    ss_neqns = ModelBaseEcon.neqns(ss)

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
        @timer r0, j0 = global_SS_RJ(xx, sd)
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

        @timer r0, j0 = global_SS_RJ(xx, sd)
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
        if any(sd.solve_var[length(model.variables) * 2 + 1:end])
            try
                global_SS_R!(r0, xx, sd)
                n1 = sum(abs2, r0)
                ssvals[sd.solve_var] .= xx
                ssvals .= update_auxvars_ss(ssvals, model)
                global_SS_R!(r0, ssvals[sd.solve_var], sd)
                n2 = sum(abs2, r0)
                if n2 < n1
                    xx .= ssvals[sd.solve_var]
                    if verbose
                        @info "    -- updated auxiliary variables"
                    end
                end
            catch
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
    if verbose && nf > tol 
        bad_eqn, bad_var = diagnose_sstate(model)
        lbva = length(bad_var)
        bad_var = join(bad_var, ",")
        vars_str = lbva > 0 ? "Unable to solve for $(lbva) variables:\n   $bad_var" : ""
        if length(vars_str) > 0
            @warn vars_str
        end
        lbeq = length(bad_eqn)
        bad_eqn = (lbeq > 10) ? (join(bad_eqn[1:10], "\n   ") * "\n   . . .") : join(bad_eqn, "\n   ")
        eqns_str = lbeq > 0 ? "System may be inconsistent. Couldn't solve $(lbeq) equations to the required accuracy:\n   $bad_eqn" : ""
        if length(eqns_str) > 0
            @warn eqns_str 
        end
    end
    xx[ abs.(xx) .< tol ] .= 0.0
    ss.values[sd.solve_var] .= xx
    ss.mask[sd.solve_var] .= true
    if run_nr
        foo = eachindex(ss.vars)[sd.solve_var]
        for v in findall(.!nr.updated)
            ss.mask[foo[v]] = false
        end
    end
    return ss.values
end
@assert precompile(sssolve!, (Model,))