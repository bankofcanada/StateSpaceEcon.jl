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
    lm = linearize!(model)
    fm = solve!(deepcopy(model), :firstorder)
    p = Plan(lm, 1U:1000U)
    xp = autoexogenize!(deepcopy(p), lm, 1U:500U)
    test_rng = first(p.range) .+ (0:500)
    Random.seed!(0xFF)
    for i = 1:30
        data = steadystatedata(lm, p)
        if i <= 5
            data[begin:1U-1, lm.variables] .+= 0.3 * rand(lm.maxlag, length(lm.variables))
        else
            data[1U.+(0:i-5), lm.shocks] .+= 0.3 * rand(1 + i - 5, length(lm.shocks))
        end
        # can we replicate stacked time?
        sol = simulate(lm, p, data, fctype=fcslope, anticipate=false)
        fsol = StateSpaceEcon.FirstOrderSolver.simulate(fm, p, data, anticipate=false)
        @test sol[test_rng] ≈ fsol[test_rng]
        # can we back out unanticipated shocks
        copyto!(data, fsol)
        data[.!xp.exogenous] .= 0
        data[begin:1U-1, :] .= fsol
        xsol = StateSpaceEcon.FirstOrderSolver.simulate(fm, xp, data, anticipate=false)
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

