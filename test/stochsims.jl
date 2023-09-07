##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

@using_example E6

@testset "stochsims" begin
    m = getE6()
    m.options.fctype = fcnatural
    @steadystate m lp = 0.6
    @steadystate m ly = 0.3
    @steadystate m lyn = 011
    clear_sstate!(m)
    sssolve!(m)
    @test issssolved(m)
    plan = Plan(m, 1U:50U)
    db_ss = steadystatedata(m, plan)
    @test db_ss ≈ simulate(m, plan, db_ss; anticipate=true)

    srng = 1U:5U

    shk_sample = [
        MVTSeries(srng, m.shocks, rand),
        MVTSeries(srng, m.shocks, rand),
        MVTSeries(srng, m.shocks, rand),
        MVTSeries(srng, m.shocks, rand)
    ]

    res = stoch_simulate(m, plan, db_ss, shk_sample)
    for (r, s) in zip(res, shk_sample)
        tmp = copy(db_ss)
        tmp .= tmp .+ s
        @test !(r ≈ simulate(m, plan, tmp))
        @test (r ≈ simulate(m, plan, tmp; anticipate=false))
    end

    # baseline that is not the steady state
    db_1 = copy(db_ss)
    shk_1 = MVTSeries(srng, m.shocks, rand)
    db_1 .= db_1 .+ shk_1

    @test simulate(m, plan, db_1; anticipate=false) ≈ stoch_simulate(m, plan, db_ss, shk_1)[1]
    @test simulate(m, plan, db_1; anticipate=false) ≈ stoch_simulate(m, plan, rawdata(db_ss), shk_1)[1]
    @test simulate(m, plan, db_1; anticipate=false) ≈ stoch_simulate(m, plan, Workspace(; pairs(db_ss)...), shk_1)[1]

    db_1 = simulate(m, plan, db_1; anticipate=true)
    res = stoch_simulate(m, plan, db_1, shk_sample; check=true)
    for (r, s) in zip(res, shk_sample)
        tmp = copy(db_1)
        tmp .= tmp .+ s
        # it's not anticipated shocks
        @test !(r ≈ simulate(m, plan, tmp; anticipate=true))
        # it's not unanticipated shocks
        @test !(r ≈ simulate(m, plan, tmp; anticipate=false))
        # it's mixed
        @test (r ≈ simulate(m, plan, db_1, plan, tmp))
    end

    # try with Workspace
    shk_w = Workspace(Symbol.(:s, eachindex(shk_sample)) .=> shk_sample)
    res_w = stoch_simulate(m, plan, db_1, shk_w)
    for i in eachindex(shk_sample)
        @test res_w[Symbol(:s, i)] ≈ res[i]
    end

end
