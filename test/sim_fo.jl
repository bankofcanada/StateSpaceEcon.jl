##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

using Random

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
    run_fo_unant_tests(M.model)
end

module R
using ModelBaseEcon
model = Model()
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
    run_fo_unant_tests(R.model)
end


@using_example E2
@testset "E2.fo.unant" begin
    run_fo_unant_tests(E2.model)
end

@using_example E3
@testset "E3.fo.unant" begin
    run_fo_unant_tests(E3.model)
end

@using_example E6
@testset "E6.fo.unant" begin
    let m = E6.model
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
    for m in (M.model, R.model)
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_shockdecomp_firstorder(m)
        m.rho = 1
        @steadystate m x = 6
        test_shockdecomp_firstorder(m, 1U:20U, fclevel)
    end
    for m in (E2.model, E3.model)
        empty!(m.sstate.constraints)
        test_shockdecomp_firstorder(m, 1U:500U)
    end
    let m = E6.model
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
    for m in (M.model, R.model)
        m.rho = 0.6
        empty!(m.sstate.constraints)
        test_initdecomp_firstorder(m)
        m.rho = 1
        @steadystate m x = 6
        test_initdecomp_firstorder(m)
    end
    for m in (E2.model, E3.model)
        empty!(m.sstate.constraints)
        test_initdecomp_firstorder(m)
    end
    let m = E6.model
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


