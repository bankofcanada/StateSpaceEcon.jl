##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2026, Bank of Canada
# All rights reserved.
##################################################################################

# The solvers reject a trial point when evaluating an equation there throws a
# `DomainError` (e.g. `log` of a negative number). Any other exception is a bug
# or an interrupt, and must reach the caller instead of being reported as an
# inadmissible point or a failed simulation.
#
# The test equations use `exc_log`, which is `log` except that, while armed, it
# throws the armed exception instead of the `DomainError`. So the current point
# can be evaluated and only the trial points of the solver fail.

const EXC_LA = StateSpaceEcon.StackedTimeSolver.LinearAlgebra

const EXC_ARMED = Ref{Any}(nothing)
const EXC_NEGATIVE = Ref(0)   # number of calls with a negative argument
function exc_log(x)
    if x < 0
        EXC_NEGATIVE[] += 1
        isnothing(EXC_ARMED[]) || throw(EXC_ARMED[])
    end
    return log(x)
end
# like `identity`, but throws the armed exception when called while updating
# auxiliary variables
function exc_aux(x)
    if !isnothing(EXC_ARMED[]) && any(fr -> fr.func === :update_auxvars, stacktrace())
        throw(EXC_ARMED[])
    end
    return x
end
function exc_armed(f, err)
    EXC_ARMED[] = err
    try
        return f()
    finally
        EXC_ARMED[] = nothing
    end
end
const EXC_BUGS = (MethodError(exc_log, (1,)), UndefVarError(:exc_undefined), ErrorException("user error"), InterruptException())
# what `@test_throws` should expect: the exact exception, except for
# `InterruptException`, whose instance (no fields) would match any exception
exc_expected(err) = err isa InterruptException ? InterruptException : err

# steady state x = 1, starting from x0
function exc_ssmodel(x0)
    m = Model()
    m.flags.ssZeroSlope = true
    @variables m x
    @equations m begin
        0 = exc_log(x[t])
    end
    @initialize m
    m.sstate.values .= [x0, 0.0]
    return m
end

# x[t] = exp(e[t] + 0.5 e[t-1] + ...), steady state x = 1
function exc_dynmodel()
    m = Model()
    @variables m x
    @shocks m e
    @equations m begin
        exc_log(x[t]) = 0.5 * exc_log(x[t-1]) + e[t]
    end
    @initialize m
    @steadystate m x = 1
    sssolve!(m)
    return m
end
struct ExcUnprintable end
Base.show(::IO, ::ExcUnprintable) = error("cannot show")
# its display is that of an object which refers back to it (the limit stops a runaway recursion)
mutable struct ExcBackref
    parent
    visits::Int
end
function Base.show(io::IO, x::ExcBackref)
    x.visits += 1
    x.visits > 8 && error("recursion")
    show(io, x.parent)
end
exc_shock(v) =(s = MVTSeries(1U:1U, (:e,), zeros); s[1U, :e] = v; s)

@testset "exceptions, steady state" begin
    SSS = StateSpaceEcon.SteadyStateSolver

    # evaluation at a given point: domain error -> inadmissible point
    m = exc_ssmodel(-1.0)
    @suppress_err begin
        @test_throws "Inadmissible point" SSS.global_SS_R!(zeros(1), [-1.0, 0.0], m)
        @test_throws "Inadmissible point" SSS.global_SS_RJ([-1.0, 0.0], m)
        @test_throws "Inadmissible point" check_sstate(m)
    end
    m = exc_ssmodel(10.0)
    sd = SSS.SolverData(m; presolve=false)
    x = fill(-1.0, sd.nvars)
    @suppress_err begin
        @test_throws "Inadmissible point" SSS.global_SS_R!(zeros(sd.neqns), x, sd)
        @test_throws "Inadmissible point" SSS.global_SS_RJ(x, sd)
    end
    @test SSS._active_SS_R!(zeros(sd.neqns), x, sd) == 1
    @test SSS._active_SS_R!(zeros(sd.neqns), fill(2.0, sd.nvars), sd) == 0
    # any other exception propagates
    for err in EXC_BUGS
        m = exc_ssmodel(-1.0)
        @test_throws exc_expected(err) exc_armed(() -> SSS.global_SS_R!(zeros(1), [-1.0, 0.0], m), err)
        @test_throws exc_expected(err) exc_armed(() -> SSS.global_SS_RJ([-1.0, 0.0], m), err)
        @test_throws exc_expected(err) exc_armed(() -> check_sstate(m), err)
        @test_throws exc_expected(err) exc_armed(() -> SSS.global_SS_R!(zeros(sd.neqns), x, sd), err)
        @test_throws exc_expected(err) exc_armed(() -> SSS.global_SS_RJ(x, sd), err)
        @test_throws exc_expected(err) exc_armed(() -> SSS._active_SS_R!(zeros(sd.neqns), x, sd), err)
    end

    # presolve (bisection): a domain error at the starting point leaves the equation unsolved
    m = exc_ssmodel(-1.0)
    m.sstate.mask[2] = true
    @test SSS.presolve_sstate!(m) == false
    for err in EXC_BUGS
        @test_throws exc_expected(err) exc_armed(() -> SSS.presolve_sstate!(m), err)
    end
    # bisection trial points
    @test_throws InterruptException SSS.bisect!(v -> throw(InterruptException()), [1.0], 1, 1.0)
    @test_throws MethodError SSS.bisect!(v -> throw(MethodError(exc_log, (1,))), [1.0], 1, 1.0)

    # Newton line search: the full step from 10 lands at 10 - 10 log(10) < 0, which
    # is rejected quietly (no "Inadmissible point" error message)
    m = exc_ssmodel(10.0)
    EXC_NEGATIVE[] = 0
    out = @capture_err begin
        @test sssolve!(m; presolve=false, nropts=Options(linesearch=true)) ≈ [1.0, 0.0]
    end
    @test EXC_NEGATIVE[] > 0
    @test !occursin("Inadmissible", out)
    for err in EXC_BUGS
        m = exc_ssmodel(10.0)
        @test_throws exc_expected(err) exc_armed(() -> sssolve!(m; presolve=false, nropts=Options(linesearch=true)), err)
    end

    # first Levenberg-Marquardt step from 5: one trial point is negative
    m = exc_ssmodel(5.0)
    EXC_NEGATIVE[] = 0
    out = @capture_err begin
        @test sssolve!(m; presolve=false, method=:lm) ≈ [1.0, 0.0]
    end
    @test EXC_NEGATIVE[] > 0
    @test !occursin("Inadmissible", out)
    for err in EXC_BUGS
        m = exc_ssmodel(5.0)
        @test_throws exc_expected(err) exc_armed(() -> sssolve!(m; presolve=false, method=:lm), err)
    end

    # update of auxiliary variables between iterations
    let m = Model()
        m.options.substitutions = true
        m.flags.ssZeroSlope = true
        @variables m x
        @equations m begin
            log(exc_aux(x[t])) = 0
        end
        @initialize m
        @test m.nauxs == 1
        m.sstate.values .= [5.0, 0.0, 0.0, 0.0]
        @test sssolve!(m; presolve=false)[1] ≈ 1.0
        for err in EXC_BUGS
            m.sstate.values .= [5.0, 0.0, 0.0, 0.0]
            @test_throws exc_expected(err) exc_armed(() -> sssolve!(m; presolve=false), err)
        end
    end
end

@testset "exceptions, simulate" begin
    m = exc_dynmodel()
    p = Plan(m, 1U:20U)

    # Newton with line search: from x = 1 the full step lands at 1 - 3 < 0
    d = steadystatedata(m, p)
    d[1U, :e] = -3.0
    EXC_NEGATIVE[] = 0
    @test simulate(m, p, d; linesearch=true)[1U:3U, :x].values ≈ exp.(-3.0 .* [1, 0.5, 0.25])
    @test EXC_NEGATIVE[] > 0
    for err in EXC_BUGS
        @test_throws exc_expected(err) exc_armed(() -> simulate(m, p, d; linesearch=true), err)
    end

    # two equations: the first one's residual shrinks before the second one throws, so the
    # damping must not judge the trial step by the partly updated residual
    m2 = Model()
    @variables m2 a b
    @shocks m2 ea eb
    @equations m2 begin
        a[t] = 0.5 * a[t-1] + ea[t]
        log(b[t]) = 0.5 * log(b[t-1]) + eb[t]
    end
    @initialize m2
    @steadystate m2 b = 1
    sssolve!(m2)
    p2 = Plan(m2, 1U:10U)
    d2 = steadystatedata(m2, p2)
    d2[1U, :ea] = 10.0
    d2[1U, :eb] = -3.0
    for kw in ((linesearch=true,), (damping=:br81,))
        r = simulate(m2, p2, d2; kw...)
        @test r[1U, :a] ≈ 10.0
        @test r[1U:3U, :b].values ≈ exp.(-3.0 .* [1, 0.5, 0.25])
    end

    # Levenberg-Marquardt: the first trial point is negative
    d = steadystatedata(m, p)
    d[1U, :e] = -1.5
    EXC_NEGATIVE[] = 0
    @test simulate(m, p, d; sim_solver=:sim_lm)[1U:3U, :x].values ≈ exp.(-1.5 .* [1, 0.5, 0.25])
    @test EXC_NEGATIVE[] > 0
    for err in EXC_BUGS
        @test_throws exc_expected(err) exc_armed(() -> simulate(m, p, d; sim_solver=:sim_lm), err)
    end
end

@testset "exceptions, stoch_simulate" begin
    m = exc_dynmodel()
    p = Plan(m, 1U:30U)   # at least 20 periods after the shock, or stoch_simulate warns
    base = steadystatedata(m, p)

    # a numerical failure marks only that realization as failed, and keeps the cause
    # (without line search, the first Newton step goes to x < 0)
    r = stoch_simulate(m, p, base, [exc_shock(-3.0), exc_shock(0.1)])
    @test r[1] isa SimFailed && r[1].info == 1U && r[1].cause isa DomainError
    @test occursin(r"in period 1U.\nCaused by: DomainError", sprint(showerror, r[1]))
    @test !isfailed(r[2]) && r[2][1U, :x] ≈ exp(0.1)
    # without a cause, the message is as before
    @test sprint(showerror, SimFailed(1U)) == "Simulation failed in period 1U."
    # a cause that cannot be displayed does not hide the failure
    @test sprint(showerror, SimFailed(1U, DomainError(ExcUnprintable()))) ==
          "Simulation failed in period 1U.\nCaused by: DomainError (cannot be displayed)"
    # displayed (as in the REPL) the same way, since stoch_simulate returns it, not throws it
    @test sprint(show, MIME("text/plain"), r[1]) == sprint(showerror, r[1])
    @test sprint(show, MIME("text/plain"), SimFailed(1U, DomainError(ExcUnprintable()))) ==
          "Simulation failed in period 1U.\nCaused by: DomainError (cannot be displayed)"
    # in a printed vector of results it keeps its compact form, also when the cause cannot be displayed
    @test sprint(show, SimFailed(1U, DomainError(-1.0, "neg"))) == "SimFailed(1U, DomainError(-1.0, \"neg\"))"
    @test occursin("SimFailed(1U, DomainError(-1.0", sprint(show, MIME("text/plain"), [SimFailed(1U, DomainError(-1.0))]))
    @test sprint(show, SimFailed(1U, DomainError(ExcUnprintable()))) == "SimFailed(1U, DomainError (cannot be displayed))"
    @test occursin("(cannot be displayed)", sprint(show, MIME("text/plain"), [SimFailed(1U, DomainError(ExcUnprintable()))]))
    # as the default display does, reset the type information and detect a circular reference
    @test sprint(show, SimFailed(true, nothing); context=:typeinfo => Bool) == "SimFailed(true, nothing)"
    let cause = ExcBackref(nothing, 0)
        ex = cause.parent = SimFailed(1U, cause)
        @test occursin("circular reference", sprint(show, ex))
        @test cause.visits == 1
    end

    # a bug or an interrupt stops the simulation
    for err in (ArgumentError("damping bug"), InterruptException())
        @test_throws exc_expected(err) stoch_simulate(m, p, base, [exc_shock(0.1)]; damping=(args...) -> throw(err))
    end
    for err in EXC_BUGS
        @test_throws exc_expected(err) exc_armed(() -> stoch_simulate(m, p, base, [exc_shock(-3.0)]), err)
    end

    # reaching maxiter
    shk = [exc_shock(0.1)]
    r = stoch_simulate(m, p, base, shk; maxiter=0, warn_maxiter=:error)
    @test r[1] isa SimFailed && r[1].info == 1U && r[1].cause isa ErrorException
    @test occursin("maxiter", sprint(showerror, r[1]))
    r = @test_logs (:warn, r"maxiter") stoch_simulate(m, p, base, shk; maxiter=0, warn_maxiter=:warn)
    @test !isfailed(r[1])
    r = @test_logs (:warn, r"maxiter") stoch_simulate(m, p, base, shk; maxiter=0, warn_maxiter=true)
    @test !isfailed(r[1])
    r = @test_logs stoch_simulate(m, p, base, shk; maxiter=0, warn_maxiter=false)
    @test !isfailed(r[1])

    # a failed factorization in one realization does not affect the next ones:
    # the Jacobian of `(x - 1 + e)^2 = 1` at x = 0 is singular for e = 1, and that
    # of e = 1 - 1e-9 is close to it, so the cached factorization would be reused
    # (with UMFPACK the next realization then failed, with Pardiso it was wrong)
    for fact in (:umfpack, :pardiso)
        fact == :pardiso && Sys.isapple() && continue   # as in runtests.jl
        m = Model()
        m.options.factorization = fact
        @variables m x
        @shocks m e
        @equations m begin
            (x[t] - 1.0 + e[t])^2 + 0.0 * x[t-1] = 1.0
        end
        @initialize m
        m.sstate.values .= 0.0
        m.sstate.mask .= true
        p = Plan(m, 1U:30U)
        base = steadystatedata(m, p)
        shock(v) = (s = MVTSeries(1U:1U, (:e,), zeros); s[1U, :e] = v; s)
        alone = stoch_simulate(m, p, base, [shock(1.0 - 1e-9)])
        r = @suppress_err stoch_simulate(m, p, base, [shock(0.1), shock(1.0), shock(1.0 - 1e-9)])
        @test !isfailed(r[1])
        @test r[2] isa SimFailed && StateSpaceEcon.StackedTimeSolver._is_numerical_failure(r[2].cause)
        @test !isfailed(alone[1]) && !isfailed(r[3]) && r[3] ≈ alone[1]
    end

    # Pardiso reports numerical and other failures with the same type
    STS = StateSpaceEcon.StackedTimeSolver
    @test STS._is_numerical_failure(DomainError(-1.0))
    @test STS._is_numerical_failure(EXC_LA.SingularException(1))
    @test STS._is_numerical_failure(Pardiso.PardisoPosDefException("Zero pivot"))
    @test STS._is_numerical_failure(Pardiso.PardisoException(STS._PARDISO_PIVOT_MESSAGE))
    @test STS._is_numerical_failure(Pardiso.PardisoException("Diagonal matrix is singular"))
    @test !STS._is_numerical_failure(Pardiso.PardisoException("Not enough memory."))
    @test !STS._is_numerical_failure(Pardiso.PardisoException("Input inconsistent."))
    @test !STS._is_numerical_failure(MethodError(exc_log, (1,)))
    @test !STS._is_numerical_failure(InterruptException())
    @test !STS._is_numerical_failure(ErrorException(STS._MAXITER_MESSAGE))
end

@testset "exceptions, EM Cholesky" begin
    chol! = StateSpaceEcon.DFMSolver._my_cholesky!
    # not positive definite: diagnostics, then the original exception
    out = @capture_out begin
        @test_logs (:error, r"Cholesky failed") @test_throws EXC_LA.PosDefException chol!([1.0 2.0; 2.0 1.0])
    end
    @test occursin("ev=", out)
    # any other exception: no diagnostics
    out = @capture_out begin
        @test_logs @test_throws InexactError chol!([4 2; 2 3])
    end
    @test isempty(out)
end

@testset "inadmissible point message" begin
    SSS = StateSpaceEcon.SteadyStateSolver
    m = Model()
    m.flags.ssZeroSlope = true
    @variables m exc_noeval
    @equations m begin
        0 = sqrt(exc_noeval[t])
    end
    @initialize m
    # the message names the variable values without assigning them in the solver module
    @suppress_err @test_throws "Inadmissible point" SSS.global_SS_RJ([-1.0, 0.0], m)
    @test !any(occursin("exc_noeval", string(n)) for n in names(SSS; all=true))
    # finite residual, infinite derivative
    @test_throws "Singular gradient" @suppress_err SSS.global_SS_RJ([0.0, 0.0], m)
    # a variable that appears twice in a constraint
    m = Model()
    m.flags.ssZeroSlope = true
    @variables m exc_twice
    @equations m begin
        exc_twice[t] = exc_twice[t-1]
    end
    @initialize m
    @steadystate m log(exc_twice) + log(exc_twice) = 0
    vsyms = only(values(m.sstate.constraints)).vsyms
    @test length(unique(vsyms)) < length(vsyms)
    m.sstate.values .= [-1.0, 0.0]
    @test_throws "Inadmissible point" @suppress_err check_sstate(m)
end
