##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

using Random
using LinearAlgebra

if !isdefined(@__MODULE__, :run_fo_unant_tests)
function run_fo_unant_tests(model)
    clear_sstate!(model)
    sssolve!(model)
    linearize!(model)
    solve!(model, solver=:stackedtime)
    solve!(model, solver=:firstorder)
    plan = Plan(model, 1U:1000U)
    xplan = autoexogenize!(deepcopy(plan), model, 1U:500U)
    test_rng = first(plan.range) .+ (0:500)
    Random.seed!(0xFF)
    for i = 1:30
        data = steadystatedata(model, plan)
        if (i <= 5) && (model.maxlag > 0)
            data[begin.+(0:model.maxlag-1), model.variables] .+= 0.3 * rand(model.maxlag, length(model.variables))
        elseif i > 5
            data[1U.+(0:i-5), model.shocks] .+= 0.3 * rand(1 + i - 5, length(model.shocks))
            if i > 10
                exog_endo!(xplan, values(model.autoexogenize), keys(model.autoexogenize), 2U:5U)
            end
        end
        # can we replicate stacked time?
        sol = simulate(model, plan, data, solver=:stackedtime, fctype=fcslope, anticipate=false)
        fsol = simulate(model, plan, data, solver=:firstorder, anticipate=false)
        @test sol[test_rng] ≈ fsol[test_rng]
        # can we back out unanticipated shocks
        copyto!(data, fsol)
        data[.!xplan.exogenous] .= 0
        data[begin:1U-1, :] .= fsol
        xsol = simulate(model, xplan, data, solver=:firstorder, anticipate=false)
        @test xsol[test_rng] ≈ fsol[test_rng]
    end
end
end

module M
using ModelBaseEcon
model = Model()
model.flags.linear = true
@variables model x
@shocks model x_shk
@parameters model rho = 0.6
@equations model begin
    x[t+1] = rho * x[t] + x_shk[t]
end
@autoexogenize model begin
    x = x_shk
end
@initialize model
end

@testset "M.fo.unant" begin
    run_fo_unant_tests(deepcopy(M.model))
end

module R
using ModelBaseEcon
model = Model()
model.flags.linear = true
@variables model begin
    @log x
    @shock x_shk
end
@parameters model rho = 0.6
@autoexogenize model begin
    x = x_shk
end
@equations model begin
    log(x[t]) = rho * log(x[t-1]) + x_shk[t]
end
@initialize model
end
@testset "R.fo.unant" begin
    run_fo_unant_tests(deepcopy(R.model))
end


@using_example E2
@testset "E2.fo.unant" begin
    run_fo_unant_tests(getE2())
end

@using_example E3
@testset "E3.fo.unant" begin
    run_fo_unant_tests(getE3())
end

@using_example E6
@testset "E6.fo.unant" begin
    let m = getE6()
        # set slopes to 0, otherwise we're not allowed to linearize
        m.p_dly = 0
        m.p_dlp = 0
        empty!(m.sstate.constraints)
        @steadystate m lp = 1.5
        @steadystate m ly = 1.1
        @steadystate m lyn = 1.1
        m
    end |> run_fo_unant_tests
end


if !isdefined(@__MODULE__, :test_shockdecomp_firstorder)
function test_shockdecomp_firstorder(m, rng=1U:20U, fctype=fcslope)
    clear_sstate!(m)
    sssolve!(m)
    # printsstate(m)
    linearize!(m)
    solve!(m, solver=:firstorder)

    p = Plan(m, rng)
    control = steadystatedata(m, p)
    data = copy(control)
    data[1U, m.shocks] .+= 1.0
    r1 = shockdecomp(m, p, data; control, solver=:stackedtime, fctype)
    r2 = shockdecomp(m, p, data; control, solver=:firstorder)

    @test simulate(m, p, data; solver=:stackedtime, fctype) ≈ r1.s
    @test simulate(m, p, data; solver=:firstorder) ≈ r2.s
    @test compare(r1, r2, quiet=true, ignoremissing=true, atol=2^10 * eps(1.0), rtol=sqrt(eps(1.0)))

    return (; r1, r2)
end
end

@testset "shkdcmp.fo" begin
    for m in (M.model, R.model)
        m = deepcopy(m)
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_shockdecomp_firstorder(m)
        m.rho = 1
        @steadystate m x = 6
        test_shockdecomp_firstorder(m, 1U:20U, fclevel)
    end
    for m in (E2.model, E3.model)
        m = deepcopy(m)
        empty!(m.sstate.constraints)
        test_shockdecomp_firstorder(m, 1U:500U)
    end
    let m = deepcopy(E6.model)
        # set slopes to 0, otherwise we're not allowed to linearize
        m.p_dly = 0
        m.p_dlp = 0
        empty!(m.sstate.constraints)
        @steadystate m lp = 1.5
        @steadystate m ly = 1.1
        @steadystate m lyn = 1.1
        test_shockdecomp_firstorder(m, 1U:100U)
    end
end

if !isdefined(@__MODULE__, :test_initdecomp_firstorder)
function test_initdecomp_firstorder(m, rng=2001Q1:2020Q4, step=max(2, length(rng) ÷ 5))
    clear_sstate!(m)
    sssolve!(m)
    # printsstate(m)
    linearize!(m)
    solve!(m, solver=:firstorder)

    Random.seed!(0xFF)

    p = Plan(m, rng)
    exog = steadystatedata(m, p)
    exog[begin:first(rng)-1, m.variables] .= rand(m.maxlag, m.nvars)
    exog[rng, m.shocks] .= randn(length(rng), m.nshks)

    res = shockdecomp(m, p, exog; solver=:firstorder)

    rs = Workspace[]
    res1 = Workspace(s=res.s[begin:first(rng)-1, :])
    for i = 0:step:length(rng)-(step-1)
        rr = rng[begin+i:begin+i+(step-1)]
        pp = Plan(m, rr)
        ee = zerodata(m, pp)
        ee[rr] .= exog
        ee[begin:first(rr)-1, :] .= res1.s
        res1 = shockdecomp(m, pp, ee; solver=:firstorder, initdecomp=res1)
        push!(rs, res1)
        @test compare(res, res1, trange=first(p.range):last(rr), atol=2^10 * eps(), quiet=true)
    end
    return rs
end
end


@testset "inidcmp.fo" begin
    for m in (M.model, R.model)
        m = deepcopy(m)
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_initdecomp_firstorder(m)
        m.rho = 1
        @steadystate m x = 6
        test_initdecomp_firstorder(m)
    end
    for m in (getE2(), getE3())
        m = deepcopy(m)
        empty!(m.sstate.constraints)
        test_initdecomp_firstorder(m)
    end
    let m = getE6()
        # set slopes to 0, otherwise we're not allowed to linearize
        m.p_dly = 0
        m.p_dlp = 0
        empty!(m.sstate.constraints)
        @steadystate m lp = 1.5
        @steadystate m ly = 1.1
        @steadystate m lyn = 1.1
        test_initdecomp_firstorder(m)
    end
end

if !isdefined(@__MODULE__, :test_initdecomp_stackedtime)
function test_initdecomp_stackedtime(m; nonlin=!m.linear, rng=2001Q1:2024Q4, fctype=fcnatural)
    solver = :stackedtime
    atol = 2^10 * eps()
    shks = filter(v -> isshock(v) || isexog(v), m.allvars)
    vars = filter(v -> !isshock(v) && !isexog(v), m.allvars)

    clear_sstate!(m)
    sssolve!(m)
    linearize!(m)
    solve!(m; solver)

    Random.seed!(0xFF)

    # test 1 - only initial conditions, no shocks
    p = Plan(m, rng)
    exog = steadystatedata(m, p)
    exog[begin:first(rng)-1, vars] = rand(m.maxlag, m.nvars)
    # zero shocks
    ref = shockdecomp(m, p, exog; solver, fctype)
    for v in vars, s in shks
        @test norm(ref.sd[v][:, s], Inf) < atol
    end

    rng1 = rng[5:end]
    p1 = Plan(m, rng1)
    exog1 = zerodata(m, p1)
    exog1[begin:first(rng1)-1, vars] = ref.s
    exog1[rng1, shks] = ref.s[rng1, shks]
    if fctype === fclevel
        exog1[last(rng1)+1:end, vars] = ref.s
    end
    res1 = shockdecomp(m, p1, exog1; solver, fctype, initdecomp=ref)
    res1a = shockdecomp(m, p1, exog1; solver, fctype)
    for v in vars, s in shks
        @test norm(res1.sd[v][:, s], Inf) < atol
        @test norm(res1a.sd[v][:, s], Inf) < atol
    end
    @test rangeof(ref) == rangeof(res1)
    @test compare(ref, res1; atol, quiet=true)
    # without initdecomp the resulting range is shorter
    # and the numbers will be identical for linear models
    @test rangeof(ref) ≠ rangeof(res1a)
    @test compare(ref, res1a; atol, quiet=true, ignoremissing=true) == !nonlin

    #test 2 - only 1 shock at a time
    for shk in shks
        exog = steadystatedata(m, p)
        exog[rng[1:4], shk] = 0.5 * randn(4)
        ref = shockdecomp(m, p, exog; solver, fctype)
        for v in vars, s in (:init, shks...)
            s === shk && continue
            @test norm(ref.sd[v][:, s], Inf) < atol
        end

        rng1 = rng[5:end]
        p1 = Plan(m, rng1)
        exog1 = zerodata(m, p1)
        exog1[begin:first(rng1)-1, vars] = ref.s
        exog1[begin:first(rng1)-1, shks] .= NaN
        exog1[rng1, shks] = ref.s[rng1, shks]
        if fctype === fclevel
            exog1[last(rng1)+1:end, vars] = ref.s
        end
        res1 = shockdecomp(m, p1, exog1; solver, fctype, initdecomp=ref)
        res1a = shockdecomp(m, p1, exog1; solver, fctype)
        for v in vars, s in (:init, shks...)
            s === shk && continue
            @test norm(res1.sd[v][:, s], Inf) < atol
            @test s == :init || norm(res1a.sd[v][:, s], Inf) < atol
        end
        @test rangeof(ref) == rangeof(res1)
        @test compare(ref, res1; atol, quiet=true)
        # without initdecomp the resulting range is shorter
        # and the numbers are totally different
        @test rangeof(ref) ≠ rangeof(res1a)
        @test compare(ref, res1a; atol, quiet=true, nans=true, ignoremissing=true) == (m.maxlag == 0)
    end

    # test 3 - all shocks and init
    p = Plan(m, rng)
    exog = steadystatedata(m, p)
    exog[begin:first(rng)-1, vars] = rand(m.maxlag, m.nvars)
    exog[rng[1:4], shks] = 0.5 * randn(4, length(shks))
    ref = shockdecomp(m, p, exog; solver, fctype)

    rng1 = rng[5:end]
    p1 = Plan(m, rng1)
    exog1 = zerodata(m, p1)
    exog1[begin:first(rng1)-1, vars] = ref.s
    exog1[begin:first(rng1)-1, shks] .= NaN   # mask shocks in initial conditions
    exog1[rng1, shks] = ref.s[rng1, shks]
    if fctype === fclevel
        exog1[last(rng1)+1:end, vars] = ref.s
    end
    res1 = shockdecomp(m, p1, exog1; solver, fctype, initdecomp=ref)
    @test rangeof(ref) == rangeof(res1)
    @test compare(ref, res1; atol, quiet=true)
    return true
end
end

@testset "inidcmp.st" begin
    for m in (M.model, R.model)
        m = deepcopy(m)
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_initdecomp_stackedtime(m)
        m.rho = 1
        @steadystate m x = 6
        test_initdecomp_stackedtime(m, fctype=fclevel)
    end
    for m in (getE2(), getE3())
        m = deepcopy(m)
        empty!(m.sstate.constraints)
        test_initdecomp_stackedtime(m)
    end
    let m = getE6()
        # set slopes to 0, otherwise we're not allowed to linearize
        m.p_dly = 0
        m.p_dlp = 0
        empty!(m.sstate.constraints)
        @steadystate m lp = 1.5
        @steadystate m ly = 1.1
        @steadystate m lyn = 1.1
        test_initdecomp_stackedtime(m)
    end
end

@testset "initdcmp.mx" begin
    # test if we're able to mix and match, i.e., can we stitch together shockdecomps ran with different solvers.

    atol = 1e-13
    rtol = 1e-10
    ignoremissing = true
    quiet = true

    m = E6.newmodel()
    m.p_dly = 0
    m.p_dlp = 0
    empty!(m.sstate.constraints)
    @steadystate m lp = 1.5
    @steadystate m ly = 1.1
    @steadystate m lyn = 1.1

    clear_sstate!(m)
    sssolve!(m)
    linearize!(m)
    solve!(m; solver=:stackedtime)
    solve!(m; solver=:firstorder)

    rng = 2001Q1:2024Q4
    fctype = fcnatural
    shks = filter(v -> isshock(v) || isexog(v), m.allvars)
    vars = filter(v -> !isshock(v) && !isexog(v), m.allvars)

    p = Plan(m, rng)
    exog = steadystatedata(m, p)
    exog[begin:first(rng)-1, vars] = rand(m.maxlag, m.nvars)
    ref_s = shockdecomp(m, p, exog; solver=:stackedtime, fctype)
    ref_f = shockdecomp(m, p, exog; solver=:firstorder)

    # we've called linearize!, so stacked time and first order must be identical
    @test compare(ref_f, ref_s; atol, rtol, ignoremissing, quiet)
    for v in vars, s in shks
        @test norm(ref_s.sd[v][:, s], Inf) < 1e-14
    end

    # stitch with the other solver
    rng1 = rng[5:end]
    p1 = Plan(m, rng1)
    exog1 = zerodata(m, p1)
    exog1[begin:first(rng1)-1, vars] = ref_f.s
    exog1[rng1, shks] = ref_f.s[rng1, shks]
    if fctype === fclevel
        exog1[last(rng1)+1:end, vars] = ref_f.s
    end

    res_fs = shockdecomp(m, p1, exog1; solver=:stackedtime, fctype, initdecomp=ref_f)
    res_ss = shockdecomp(m, p1, exog1; solver=:stackedtime, fctype, initdecomp=ref_s)
    res_sf = shockdecomp(m, p1, exog1; solver=:firstorder, initdecomp=ref_s)
    res_ff = shockdecomp(m, p1, exog1; solver=:firstorder, initdecomp=ref_f)

    # res_fs.sd.:($var).term has NaN in the beginning
    foreach(values(res_fs.sd)) do x
        replace!(x -> isnan(x) ? 0 : x, x)
    end

    @test compare(res_fs, res_ss; atol, rtol, ignoremissing, quiet)
    @test compare(res_ss, res_sf; atol, rtol, ignoremissing, quiet)
    @test compare(res_sf, res_ff; atol, rtol, ignoremissing, quiet)
    @test compare(res_ff, ref_s; atol, rtol, ignoremissing, quiet)

end

@testset "fo.ant.swap" begin
    # anticipated first-order simulation with a swapped plan runs the
    # stacked-time solver on the linearized model (it used to call itself)
    m = getE3()
    clear_sstate!(m)
    sssolve!(m)
    solve!(m, solver=:firstorder)
    p = Plan(m, 1U:20U)
    exogenize!(p, :pinf, 3U)
    endogenize!(p, :pinf_shk, 3U)
    data = steadystatedata(m, p)
    data[3U, :pinf] += 0.1
    fo = @test_logs (:warn, r"linearized stacked-time") simulate(m, p, data; solver=:firstorder, anticipate=true)
    @test fo[3U, :pinf] ≈ data[3U, :pinf]
    @test fo ≈ simulate(m, p, data; solver=:stackedtime, anticipate=true, variant=:linearize)
    @test !(fo ≈ data)
    # non-zero steady state, linearized for data in deviations: the result is in levels
    m = Model()
    @variables m y
    @shocks m e
    @equations m begin
        y[t] = 0.5 * y[t-1] + 1 + e[t]
    end
    @initialize m
    sssolve!(m)
    linearize!(m; deviation=true)
    solve!(m, solver=:firstorder)
    p = Plan(m, 1U:5U)
    exogenize!(p, :y, 2U)
    endogenize!(p, :e, 2U)
    data = steadystatedata(m, p)
    data[2U, :y] += 0.1
    fo = @test_logs (:warn, r"linearized stacked-time") simulate(m, p, data; solver=:firstorder, anticipate=true)
    @test fo[1U, :y] ≈ 2
    @test fo[2U, :y] ≈ 2.1
    @test fo[2U, :e] ≈ 0.1
    # the model's linearization is unchanged
    @test ModelBaseEcon.getevaldata(m, :linearize).deviation
    # linearized about a point other than the steady state: the fallback uses that linearization
    m = Model()
    @variables m y
    @shocks m e
    @equations m begin
        y[t] = 0.1 * y[t-1]^2 + 1 + e[t]
    end
    @initialize m
    sssolve!(m)
    y0 = m.sstate.y.level
    custom = deepcopy(m.sstate)
    custom.y.level = 2.0
    linearize!(m; sstate=custom)
    solve!(m, solver=:firstorder)
    lmed = ModelBaseEcon.getevaldata(m, :linearize)
    variant = m.options.variant
    p = Plan(m, 1U:5U)
    exogenize!(p, :y, 2U)
    endogenize!(p, :e, 2U)
    data = steadystatedata(m, p)
    data[2U, :y] = 2.2
    fo = @test_logs (:warn, r"linearized stacked-time") simulate(m, p, data; solver=:firstorder, anticipate=true)
    # about y = 2 the model is y[t] = 0.4 * y[t-1] + 0.6 + e[t]
    @test fo[2U, :e] ≈ 2.2 - 0.4 * (0.4 * y0 + 0.6) - 0.6
    @test fo[3U, :y] ≈ 0.4 * 2.2 + 0.6
    @test ModelBaseEcon.getevaldata(m, :linearize) === lmed
    @test m.options.variant == variant
    # a `variant` passed by the caller does not make the fallback run the nonlinear model
    fo2 = @test_logs (:warn, r"linearized stacked-time") simulate(m, p, data; solver=:firstorder,
        anticipate=true, variant=:default)
    @test fo2 ≈ fo
    @test m.options.variant == variant
end
