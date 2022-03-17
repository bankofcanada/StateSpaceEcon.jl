##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

using StateSpaceEcon
using TimeSeriesEcon
using ModelBaseEcon

@info "Loading Model examples. Might take some time to pre-compile."
@using_example E1
@using_example E2
@using_example E3
@using_example E3nl
@using_example E6
@using_example E7
@using_example E7A
@using_example S1
@using_example S2

using Test
using Suppressor

@testset "1dsolvers" begin
    # f(x) = (x-2)*(x-3) = a x^2 + b x + c with vals = [a, x, b, c]
    let f(v) = v[1] * v[2]^2 + v[3] * v[2] + v[4],
        fdf(v) = (f(v), [v[2]^2, v[1] * v[2] * 2 + v[3], v[2], 1.0]),
        vals = [1.0, NaN, -5.0, 6.0]

        vals[2] = 0.0
        @test StateSpaceEcon.SteadyStateSolver.newton1!(fdf, vals, 2; tol = eps(), maxiter = 8)
        @test vals ≈ [1.0, 2.0, -5.0, 6.0] atol = 1e3 * eps()

        vals[2] = 6.0
        @test StateSpaceEcon.SteadyStateSolver.newton1!(fdf, vals, 2; tol = eps(), maxiter = 8)
        @test vals ≈ [1.0, 3.0, -5.0, 6.0] atol = 1e3 * eps()

        vals[2] = 0.0
        @test StateSpaceEcon.SteadyStateSolver.bisect!(f, vals, 2, fdf(vals)[2][2]; tol = eps())
        @test vals ≈ [1.0, 2.0, -5.0, 6.0] atol = 1e3 * eps()

        vals[2] = 6.0
        @test StateSpaceEcon.SteadyStateSolver.bisect!(f, vals, 2, fdf(vals)[2][2]; tol = eps())
        @test vals ≈ [1.0, 3.0, -5.0, 6.0] atol = 1e3 * eps()
    end
end

@testset "Plans" begin
    m = E1.model
    p = Plan(m, 1:3)
    @test first(p.range) == 0U
    @test last(p.range) == 4U
    out = @capture_out print(p)
    @test length(split(out, '\n')) == 2
    @test p[1] == [:y_shk]
    @test p[1U] == [:y_shk]
    endogenize!(p, :y_shk, 1U)
    @test isempty(p[1U])
    endogenize!(p, :y_shk, 1U:3U)
    exogenize!(p, :y, 2U)
    exogenize!(p, :y, 4U)
    # make sure indexing with integers works as well
    @test p[0U] == p[1] == [:y_shk]
    @test p[1U] == p[2] == []
    @test p[2U] == p[3] == [:y]
    @test p[3U] == p[4] == []
    @test p[4U] == p[5] == [:y, :y_shk]
    out = @capture_out print(p)
    @test length(split(out, '\n')) == 6
    out = @capture_out print(IOContext(stdout, :displaysize => (7, 80)), p)
    @test length(split(out, '\n')) == 4
    @test length(split(out, '⋮')) == 2
    let p = Plan(m, 2000Q1:2020Q4)
        endogenize!(p, shocks(m), 2000Q1:2002Q4)
        @test isempty(p[2000Q1])
        out = @capture_out print(p)
        length(split(out, "\n")) == 4
    end
    let p = Plan(2000Q1:2010Q4, (a = 1, b = 2, c = 3), falses(44, 3))
        exogenize!(p, :a, p.range)
        exogenize!(p, :b, 2001Q1:2006Q1)
        exogenize!(p, :c, 2006Q1:2009Q4)

        pio = IOBuffer()
        exportplan(pio, p)
        seek(pio, 0)
        q = importplan(pio)
        @test p == q
    end

end

# include("simdatatests.jl")
include("sstests.jl")

@testset "misc" begin
    m = E3.model
    sim = m.maxlag .+ (1:10)
    p = Plan(m, sim)

    # random data
    d1 = zeroworkspace(m, p)
    d = zeroworkspace(m, p)
    for v in keys(d1)
        @test d[v] == d1[v]
    end
    for v in values(d)
        v .= rand(Float64, size(v))
    end

    @test workspace2array(d1, m.allvars) == zeroarray(m, p)
    @test workspace2array(d1, m.allvars) == rawdata(zerodata(m, p))

    @test size(workspace2array(d, [:pinf, :ygap])) == (length(p.range), 2)
    @test size(workspace2array(d, ["pinf", "ygap"])) == (length(p.range), 2)
    @test size(workspace2array(d, m.variables)) == (length(p.range), 3)
    a = workspace2array(d, m.allvars)
    @test size(a) == (length(p.range), 6)
    @test a == hcat((d[v] for v in m.allvars)...)

    # error variable missing from dictionary
    @test_throws KeyError workspace2array(d, [:pinf, :ygap, :nosuchvar])
    # error out of range
    @test_throws BoundsError workspace2array(d, [:pinf, :ygap], 10U:20U)

    # warning variables with different ranges
    d.wrong_var = TSeries(3U, rand(10))
    b = workspace2array(d, [:pinf, :wrong_var])
    @test size(b) == (10, 2)
    @test b[:, 1] == d.pinf[3U:12U].values
    @test b[:, 2] == d.wrong_var[3U:12U].values

    s = workspace2data(d, m.allvars)
    @test all(s .== a)

    sa = data2array(s)  # copy=false, so s and sa point to the same matrix
    @test all(sa .== s)
    s.pinf[3U:5U] = 3:5
    @test all(sa .== s)

    sd = data2workspace(s)
    @test Set(keys(sd)) == Set(colnames(s))

    as = array2data(a, m.allvars, p.range)
    @test all(as .== a)
    a[1, 2] = 2.5
    @test all(as .== a)
    as = array2data(a, m.allvars, first(p.range), copy = true)
    @test all(as .== a)
    a[1, 2] = 3.0
    @test !all(as .== a)
    as[1, 2] = 3.0
    @test all(as .== a)

    ad = array2workspace(a, m.allvars, first(p.range))
    @test length(ad) == size(a, 2)
    @test all(ad[v].values == a[:, i] for (i, v) in enumerate(m.allvars))
end

@testset "overlay" begin
    t1 = overlay(TSeries(3U, 3ones(2)), TSeries(1U, ones(6)))
    @test t1 == TSeries(1U, [1, 1, 3, 3, 1, 1])
    t1 = overlay(TSeries(4U, 5ones(5)), t1)
    @test t1 == TSeries(1U, [1, 1, 3, 5, 5, 5, 5, 5])
end

include("simtests.jl")
include("logsimtests.jl")

# include("shockdecomp.jl")

@testset "misc" begin
    io = IOBuffer()
    printmatrix(io, rand(3, 4))
    seekstart(io)
    @test length(readlines(io)) == 3
end

include("dynss.jl")
