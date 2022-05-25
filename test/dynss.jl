##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

@testset "dynss" begin
    m = deepcopy(S1.model)
    # reset parameters
    m.α = 0.5
    m.β = 0.6

    for i = 1:10
        m.q = q = 2 + 4 * rand()
        m.a_ss = a_ss = 2 + 10 * rand()
        @test begin
            clear_sstate!(m)
            sssolve!(m)
            check_sstate(m) == 0
        end
        @test issssolved(m)
        @test m.sstate.a.level ≈ a_ss && m.sstate.a.slope == 0
        @test m.sstate.b.level ≈ a_ss / (1 + q) && m.sstate.b.slope == 0
        @test m.sstate.c.level ≈ q * a_ss / (1 + q) && m.sstate.c.slope == 0
    end

    m.q = 2
    m.a_ss = 3
    clear_sstate!(m)
    sssolve!(m)
    @test 0 == check_sstate(m)
    @test issssolved(m)

    simrng = 2000Q1:2020Q1
    p = Plan(m, simrng)

    # check we replicate the steady state solution
    ss = steadystatedata(m, p)
    @test ss ≈ simulate(m, p, ss)

    # check an irf converges to steady state
    exog = zerodata(m, p)
    exog[begin:simrng[1]-1, m.variables] .= 4.3
    exog[simrng[1:4], m.shocks] .= 0.6
    irf = simulate(m, p, exog)
    @test irf[end-3:end, :] ≈ ss[end-3:end, :]
end

@testset "dynss2" begin
    m = deepcopy(S2.model)


    m.α = 0.4
    m.x_ss = 3

    clear_sstate!(m)
    sssolve!(m)
    @test 0 == check_sstate(m)
    @test issssolved(m)
    @test m.sstate.x.level ≈ m.x_ss && m.sstate.x.slope == exp(0)
    @test m.sstate.y.level ≈ 2m.x_ss && m.sstate.y.slope == 0

    simrng = 2000Q1:2015Q1
    p = Plan(m, simrng)

    # check we replicate the steady state solution
    ss = steadystatedata(m, p)
    @test ss ≈ simulate(m, p, ss)

    # check an irf converges to steady state
    exog = zerodata(m, p)
    exog[begin:first(simrng)-1, m.variables] .= (m.x_ss + 0.5) .* [2 1]
    exog[simrng[1:4], m.shocks] .= 0.2
    irf = simulate(m, p, exog)

    @test irf[end-3:end, :] ≈ ss[end-3:end, :]

end
