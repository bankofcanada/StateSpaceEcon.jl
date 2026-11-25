##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2026, Bank of Canada
# All rights reserved.
##################################################################################

# Build a model and solve it in the same call, i.e. without returning to top level
# in between. The model's functions are created by `eval`, so the solvers can only
# see them if the public entry points cross into the latest world.
#
# ModelBaseEcon reuses the functions of identical equations in the same module, and
# functions created by an earlier call are visible to later ones. So every build
# multiplies its equations by a different factor `k`: same solution, new functions.
const WORLD_AGE_K = Ref(rand(1:10^6) * 1000)
function world_age_build()
    k = (WORLD_AGE_K[] += 1)
    m = Model()
    @variables m a b
    @shocks m sa
    @parameters m p = 0.5 q = 2.0
    @equations m begin
        $k * a[t] = $k * (p * a[t-1] + sa[t])
        $k * log(b[t]) = $k * (q * a[t] + 0.1 * a[t+1])
    end
    @initialize m
    @steadystate m $k * b = $k
    return m
end
world_age_shock(m, p) = (d = steadystatedata(m, p); d[2020Q1, :sa] = 0.1; d)
# solves the steady state of a fresh model; the dynamic entry points are tested on it
world_age_solved() = (m = world_age_build(); Base.invokelatest(sssolve!, m); m)

function world_age_steadystate()
    sspt = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]   # (level, slope) of a, b, sa
    m = world_age_build()
    presolved = (StateSpaceEcon.SteadyStateSolver.presolve_sstate!(m), m.sstate.mask[3], m.sstate.values[3])
    m = world_age_build()
    clear_sstate!(m)
    cleared = (m.sstate.mask[3], m.sstate.values[3])
    m = world_age_build()
    initial_sstate!(m, [0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    initial = (m.sstate.mask[3], m.sstate.values[3])
    m = world_age_build()
    R = fill(NaN, 5)
    StateSpaceEcon.SteadyStateSolver.global_SS_R!(R, sspt, m)
    m = world_age_build()
    RJ, _ = StateSpaceEcon.SteadyStateSolver.global_SS_RJ(sspt, m)
    m = world_age_build()
    m.sstate.values .= sspt
    nbad = check_sstate(m)
    m = world_age_build()
    m.sstate.values .= sspt
    bad_eqns, _ = diagnose_sstate(m)
    m = world_age_build()
    ss = copy(sssolve!(m))
    return (; presolved, cleared, initial, R, RJ, nbad, bad_eqns, ss)
end

function world_age_simulate()
    p = Plan(world_age_build(), 2020Q1:2020Q4)
    m = world_age_solved()
    st = simulate(m, p, world_age_shock(m, p))[:a].values
    m = world_age_solved()
    mix = simulate(m, p, world_age_shock(m, p), p, world_age_shock(m, p))[:a].values
    m = world_age_solved()
    sd = shockdecomp(m, p, world_age_shock(m, p))
    m = world_age_solved()
    stoch = stoch_simulate(m, p, steadystatedata(m, p), [world_age_shock(m, p)[2020Q1:2020Q4, m.shocks]])
    m = world_age_solved()
    solve!(m; solver=:firstorder)
    fo = simulate(m, p, world_age_shock(m, p); solver=:firstorder)[:a].values
    return (; st, mix, sd, stoch, fo)
end

@testset "build and solve in one call" begin
    r = world_age_steadystate()
    @test r.presolved == (true, true, 1.0)   # b = 1 from the constraint
    @test r.cleared == (true, 1.0)
    @test r.initial == (true, 1.0)
    @test r.R ≈ zeros(5) atol = 1e-12
    @test r.RJ ≈ zeros(5) atol = 1e-12
    @test r.nbad == 0
    @test isempty(r.bad_eqns)
    # the wrappers keep the return types the bodies infer
    @test Base.return_types(diagnose_sstate, (Model,)) == [Tuple{Tuple{Vararg{String}},Vector{Symbol}}]
    @test r.ss ≈ [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] atol = 1e-12

    s = world_age_simulate()
    a = [0.0, 0.1, 0.05, 0.025, 0.0125]   # 2019Q4:2020Q4
    @test s.st[1:5] ≈ a
    @test s.mix[1:5] ≈ a
    @test s.sd.s[:a].values[1:5] ≈ a
    @test s.stoch[1][:a].values[1:5] ≈ a
    @test s.fo[1:5] ≈ a
end
