##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

using Random
using LinearAlgebra

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
    run_fo_unant_tests(deepcopy(E2.model))
end

@using_example E3
@testset "E3.fo.unant" begin
    run_fo_unant_tests(deepcopy(E3.model))
end

@using_example E6
@testset "E6.fo.unant" begin
    let m = deepcopy(E6.model)
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

@testset "shkdcmp.fo" begin
    for m in (deepcopy(M.model), deepcopy(R.model))
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_shockdecomp_firstorder(m)
        m.rho = 1
        @steadystate m x = 6
        test_shockdecomp_firstorder(m, 1U:20U, fclevel)
    end
    for m in (deepcopy(E2.model), deepcopy(E3.model))
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


@testset "inidcmp.fo" begin
    for m in (deepcopy(M.model), deepcopy(R.model))
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_initdecomp_firstorder(m)
        m.rho = 1
        @steadystate m x = 6
        test_initdecomp_firstorder(m)
    end
    for m in (deepcopy(E2.model), deepcopy(E3.model))
        empty!(m.sstate.constraints)
        test_initdecomp_firstorder(m)
    end
    let m = deepcopy(E6.model)
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
        exog1[last(rng1)+1:end,vars] = ref.s
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
    @test compare(ref, res1a; atol, quiet=true) == !nonlin

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
            exog1[last(rng1)+1:end,vars] = ref.s
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
        @test compare(ref, res1a; atol, quiet=true, nans=true) == (m.maxlag == 0)
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
        exog1[last(rng1)+1:end,vars] = ref.s
    end
    res1 = shockdecomp(m, p1, exog1; solver, fctype, initdecomp=ref)
    @test rangeof(ref) == rangeof(res1)
    @test compare(ref, res1; atol, quiet=true)
    return true
end

@testset "inidcmp.st" begin
    for m in (deepcopy(M.model), deepcopy(R.model))
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_initdecomp_stackedtime(m)
        m.rho = 1
        @steadystate m x = 6
        test_initdecomp_stackedtime(m, fctype=fclevel)
    end
    for m in (deepcopy(E2.model), deepcopy(E3.model))
        empty!(m.sstate.constraints)
        test_initdecomp_stackedtime(m)
    end
    let m = deepcopy(E6.model)
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


