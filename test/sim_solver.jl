##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

rng = Random.default_rng()
@testset "sim_solver" begin
    m = getE7A()
    @test begin
        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        clear_sstate!(m)
        sssolve!(m)
        issssolved(m) && (0 == check_sstate(m))
    end
    p01 = Plan(m, 1:100)
    data01 = steadystatedata(m, p01)
    data01[begin:0U, m.variables] .+= randn(rng, m.maxlag, m.nvars)
    data01[1U:10U, m.shocks] .+= randn(rng, 10, m.nshks)

    res01 = []
    for sim_solver in (:sim_nr, :sim_lm, :sim_gn)
        tmp = simulate(m, p01, data01; warn_maxiter=:warn, sim_solver, verbose=false, fctype=fcnatural)
        push!(res01, tmp)
    end

    @test @compare res01[1] res01[2] quiet
    @test @compare res01[1] res01[3] quiet
    @test @compare res01[2] res01[3] quiet
end

@testset "damping" begin

    m = getE7A()
    @test begin
        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        clear_sstate!(m)
        sssolve!(m)
        issssolved(m) && (0 == check_sstate(m))
    end

    Random.seed!(rng, 0o042007)

    rngsim = 2024Q1:2049Q4
    p = Plan(m, rngsim)
    exog = steadystatedata(m, p)
    exog[firstdate(p).+(0:m.maxlag-1), m.variables] .*= exp.(0.5 * randn(rng, m.maxlag, m.nvars))
    exog[rngsim[1:8], m.shocks] .= 0.8 * randn(rng, 8, m.nshks)

    sim = nothing
    out = @capture_err begin
        sim = simulate(m, p, exog, verbose=true, fctype=fcnatural)
    end

    sim1 = simulate(m, p, exog, verbose=false, fctype=fcnatural, sim_solver=:sim_lm)
    @test sim ≈ sim1
    sim1 = simulate(m, p, exog, verbose=false, fctype=fcnatural, sim_solver=:sim_gn)
    @test sim ≈ sim1

    sim1 = nothing
    out1 = @capture_err sim1 = simulate(m, p, exog, verbose=true, fctype=fcnatural, linesearch=true)
    @test sim ≈ sim1
    @test out == out1

    out1 = @capture_err sim1 = simulate(m, p, exog, verbose=true, fctype=fcnatural, damping=0.8)
    @test sim ≈ sim1
    lines = split(out1, "\n"; keepempty=false)
    # foreach(println, lines)
    for (i, line) in enumerate(lines)
        if 1 < i < length(lines)
            @test occursin("λ = 0.8", line)
        end
    end

    out1 = @capture_err sim1 = simulate(m, p, exog, verbose=true, fctype=fcnatural, damping=[0.7, 0.8, 1])
    @test sim ≈ sim1
    lines = split(out1, "\n"; keepempty=false)
    # foreach(println, lines)
    for (i, line) in enumerate(lines)
        i == 2 && (@test occursin("λ = 0.7", line); continue)
        i == 3 && (@test occursin("λ = 0.8", line); continue)
        1 < i < length(lines) && (@test occursin("λ = 1.0", line); continue)
    end

    out1 = @capture_err sim1 = simulate(m, p, exog, verbose=true, fctype=fcnatural, damping=:armijo)
    @test sim ≈ sim1
    @test out == out1

    out1 = @capture_err sim1 = simulate(m, p, exog, verbose=true, fctype=fcnatural, damping=(:armijo, :sigma => 0.3, :alpha => 0.1))
    @test sim ≈ sim1
    @test out == out1

    out1 = @capture_err sim1 = simulate(m, p, exog, verbose=true, fctype=fcnatural, damping=:br81)
    @test sim ≈ sim1
    @test out == out1

    # try a more difficult test where damping is actually necessary
    @test begin
        empty!(m.sstate.constraints)
        @steadystate m linv = 1
        @steadystate m lc = 7
        clear_sstate!(m)
        sssolve!(m)
        issssolved(m) && (0 == check_sstate(m))
    end
    exog = steadystatedata(m, p)
    Random.seed!(rng, 0o007)
    exog[firstdate(p).+(0:m.maxlag-1), m.variables] .*= exp.(17 * randn(rng, m.maxlag, m.nvars))
    exog[rngsim[1:8], m.shocks] .= 15 * randn(rng, 8, m.nshks)

    # outp = @capture_err @test_throws SingularException simulate(m, p, exog, verbose=true, fctype=fcnatural)
    outp = @capture_err sim = simulate(m, p, exog, verbose=true, fctype=fcnatural)
    lines = split(outp, "\n"; keepempty=false)
    # foreach(println, lines)
    for (i, line) in enumerate(lines)
        if 1 < i < length(lines)
            @test occursin("λ = 1.0", line)
        end
    end
    niter = length(lines) - 2

    sim2 = nothing
    outg = @capture_err sim2 = simulate(m, p, exog, verbose=true, fctype=fcnatural, damping=:br81)
    lines = split(outg, "\n"; keepempty=false)
    # foreach(println, lines)
    for (i, line) in enumerate(lines)
        if 1 < i < length(lines)
            @test !occursin("λ = 1.0", line)
            match1 = match(r".*λ = (.*)", line)
            @test !isnothing(match1) && 0 < parse(Float64, match1.captures[1]) < 1
        end
    end
    @test length(lines) - 2 >= niter
    @test sim ≈ sim2

    sim3 = nothing
    outg = @capture_err sim3 = simulate(m, p, exog, verbose=true, fctype=fcnatural, damping=(:br81, :delta => 0.8))
    lines = split(outg, "\n"; keepempty=false)
    # foreach(println, lines)
    for (i, line) in enumerate(lines)
        if 1 < i < length(lines)
            @test !occursin("λ = 1.0", line)
            m = match(r".*λ = (.*)", line)
            @test !isnothing(m) && 0 < parse(Float64, m.captures[1]) < 1
        end
    end
    @test length(lines) - 2 >= niter
    @test sim ≈ sim3

end





