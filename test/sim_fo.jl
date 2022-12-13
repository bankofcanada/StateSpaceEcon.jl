##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

using Random

@using_example E2
@testset "E2.fo.unant" begin
    m = deepcopy(E2.model)
    clear_sstate!(m)
    sssolve!(m)
    # printsstate(m)
    p = Plan(m, 1U:100U)
    test_rng = first(p.range) .+ (0:50)
    for j = 1:10
        data = steadystatedata(m, p)
        Random.seed!(0xFF)
        if j > 5
            data[0U.+(0:j-5), m.shocks] .+= 0.3 * rand(1 + j - 5, 3)
        else
            data[0U, m.variables] .+= 0.3 * rand(3)
        end
        sol = simulate(m, p, data, fctype=fcslope, anticipate=false)
        fm = solve!(deepcopy(m), :firstorder)
        sol_fo = StateSpaceEcon.FirstOrderSolver.simulate(fm, p, data, anticipate=false)
        @test sol_fo[test_rng] ≈ sol[test_rng] atol = 1e-11
    end
end

@using_example E3
@testset "E3.fo.unant" begin
    m = deepcopy(E3.model)
    clear_sstate!(m)
    sssolve!(m)
    # printsstate(m)
    p = Plan(m, 1U:100U)
    test_rng = first(p.range) .+ (0:50)
    for j = 1:10
        data = steadystatedata(m, p)
        Random.seed!(0xFF)
        if j > 5
            data[0U.+(0:j-5), m.shocks] .+= 0.3 * rand(1 + j - 5, 3)
        else
            data[0U, m.variables] .+= 0.3 * rand(3)
        end
        sol = simulate(m, p, data, fctype=fcslope, anticipate=false)
        fm = solve!(deepcopy(m), :firstorder)
        sol_fo = StateSpaceEcon.FirstOrderSolver.simulate(fm, p, data, anticipate=false)
        @test sol_fo[test_rng] ≈ sol[test_rng] atol = 1e-11
    end
end


