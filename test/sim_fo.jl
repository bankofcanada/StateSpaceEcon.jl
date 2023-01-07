##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
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
        if i <= 5
            data[begin:1U-1, model.variables] .+= 0.3 * rand(model.maxlag, length(model.variables))
        else
            data[1U.+(0:i-5), model.shocks] .+= 0.3 * rand(1 + i - 5, length(model.shocks))
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


@using_example E2
@testset "E2.fo.unant" begin
    run_fo_unant_tests(deepcopy(E2.model))
end

@using_example E3
@testset "E3.fo.unant" begin
    run_fo_unant_tests(deepcopy(E3.model))
end

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

