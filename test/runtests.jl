##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
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

getE1() = E1.newmodel()
getE2() = E2.newmodel()
getE3() = E3.newmodel()
getE3nl() = E3nl.newmodel()
getE6() = E6.newmodel()
getE7() = E7.newmodel()
getE7A() = E7A.newmodel()
getS1() = S1.newmodel()
getS2() = S2.newmodel()

using Test
using Random
using Suppressor
import Pardiso

@testset "1dsolvers" begin
    # f(x) = (x-2)*(x-3) = a x^2 + b x + c with vals = [a, x, b, c]
    let f(v) = v[1] * v[2]^2 + v[3] * v[2] + v[4],
        fdf(v) = (f(v), [v[2]^2, v[1] * v[2] * 2 + v[3], v[2], 1.0]),
        vals = [1.0, NaN, -5.0, 6.0]

        vals[2] = 0.0
        @test StateSpaceEcon.SteadyStateSolver.newton1!(fdf, vals, 2; tol=eps(), maxiter=8)
        @test vals ≈ [1.0, 2.0, -5.0, 6.0] atol = 1e3 * eps()

        vals[2] = 6.0
        @test StateSpaceEcon.SteadyStateSolver.newton1!(fdf, vals, 2; tol=eps(), maxiter=8)
        @test vals ≈ [1.0, 3.0, -5.0, 6.0] atol = 1e3 * eps()

        vals[2] = 0.0
        @test StateSpaceEcon.SteadyStateSolver.bisect!(f, vals, 2, fdf(vals)[2][2]; tol=eps())
        @test vals ≈ [1.0, 2.0, -5.0, 6.0] atol = 1e3 * eps()

        vals[2] = 6.0
        @test StateSpaceEcon.SteadyStateSolver.bisect!(f, vals, 2, fdf(vals)[2][2]; tol=eps())
        @test vals ≈ [1.0, 3.0, -5.0, 6.0] atol = 1e3 * eps()
    end
end

@testset "Plans" begin
    m = getE1()
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
    let p = Plan(2000Q1:2010Q4, (a=1, b=2, c=3), falses(44, 3))
        exogenize!(p, :a, p.range)
        exogenize!(p, :b, 2001Q1:2006Q1)
        exogenize!(p, :c, 2006Q1:2009Q4)

        pio = IOBuffer()
        exportplan(pio, p)
        seekstart(pio)
        q = importplan(pio)
        @test p == q
    end

end

@testset "compare_plans" begin
    let m = Model()
        m = Model()
        @variables m a b c
        @parameters m pa = 0.9 pb = -0.3 ass = 0.3 bss = 0.4
        @shocks m as bs
        @autoexogenize m begin
            a = as
            b = bs
        end
        @equations m begin
            c[t] = sqrt(b[t]^2 + a[t]^2)
            a[t] - ass = pa * (a[t-1] - ass) + as[t]
            b[t] - bss = pb * (b[t-1] - bss) + bs[t]
        end
        @initialize m

        p = Plan(m, 1U:10U)
        autoexogenize!(p, m, 2U:3U)
        q = copy(p)
        exog_endo!(q, m.a, m.as, 2U:6U)
        exog_endo!(p, m.b, m.bs, 5U:8U)

        m1 = deepcopy(m)
        deleteat!(m1.variables, 2)
        push!(m1.variables, :beta)
        r = Plan(m1, 8U:15U)
        exogenize!(r, :beta, 10U:13U)

        begin
            io = IOBuffer()
            compare_plans(io, p, q)
            seek(io, 0)
            @test read(io, String) == "\nSame range: 0U:10U\nSame variables.\n(X) = Exogenous, (~) = Endogenous, (.) = Missing:\n  NAME   0U:1U    2U:3U    4U:4U    5U:6U    7U:8U   9U:10U \n     a    ~ ~      X X      ~ X      ~ X      ~ ~      ~ ~  \n     b    ~ ~      X X      ~ ~      X ~      X ~      ~ ~  \n     c    ~ ~      ~ ~      ~ ~      ~ ~      ~ ~      ~ ~  \n    as    X X      ~ ~      X ~      X ~      X X      X X  \n    bs    X X      ~ ~      X X      ~ X      ~ X      X X  \n"
        end
        begin
            io = IOBuffer()
            compare_plans(io, p, q; pagelines=3)
            seek(io, 0)
            @test read(io, String) == "\nSame range: 0U:10U\nSame variables.\n(X) = Exogenous, (~) = Endogenous, (.) = Missing:\n  NAME   0U:1U    2U:3U    4U:4U    5U:6U    7U:8U   9U:10U \n     a    ~ ~      X X      ~ X      ~ X      ~ ~      ~ ~  \n     b    ~ ~      X X      ~ ~      X ~      X ~      ~ ~  \n     c    ~ ~      ~ ~      ~ ~      ~ ~      ~ ~      ~ ~  \n\n  NAME   0U:1U    2U:3U    4U:4U    5U:6U    7U:8U   9U:10U \n\n    as    X X      ~ ~      X ~      X ~      X X      X X  \n    bs    X X      ~ ~      X X      ~ X      ~ X      X X  \n"
        end
        begin
            # non-matching plan ranges/varshocks

            # print to io buffer
            io = IOBuffer()
            compare_plans(io, p, r; alphabetical=true)
            seek(io, 0)
            @test read(io, String) == "\nRange  left: 0U:10U\nRange right: 7U:15U\nVariables only in left plan: [:b]\nVariables only in right plan: [:beta]\n4 common variables.\n(X) = Exogenous, (~) = Endogenous, (.) = Missing:\n  NAME   0U:1U    2U:3U    4U:4U    5U:6U    7U:8U    9U:9U   10U:10U  11U:13U  14U:15U\n     a    ~ .      X .      ~ .      ~ .      ~ ~      ~ ~      ~ ~      . ~      . ~  \n    as    X .      ~ .      X .      X .      X X      X X      X X      . X      . X  \n     b    ~ .      X .      ~ .      X .      X .      ~ .      ~ .      . .      . .  \n  beta    . .      . .      . .      . .      . ~      . ~      . X      . X      . ~  \n    bs    X .      ~ .      X .      ~ .      ~ X      X X      X X      . X      . X  \n     c    ~ .      ~ .      ~ .      ~ .      ~ ~      ~ ~      ~ ~      . ~      . ~  \n"
            
            # print only rows with differences
            io = IOBuffer()
            compare_plans(io, p, r; alphabetical=true, diff=true)
            seek(io, 0)
            @test read(io, String) == "\nRange  left: 0U:10U\nRange right: 7U:15U\nVariables only in left plan: [:b]\nVariables only in right plan: [:beta]\n4 common variables.\n(X) = Exogenous, (~) = Endogenous, (.) = Missing:\n  NAME   0U:1U    2U:3U    4U:4U    5U:6U    7U:8U    9U:9U   10U:10U  11U:13U  14U:15U\n     b    ~ .      X .      ~ .      X .      X .      ~ .      ~ .      . .      . .  \n  beta    . .      . .      . .      . .      . ~      . ~      . X      . X      . ~  \n    bs    X .      ~ .      X .      ~ .      ~ X      X X      X X      . X      . X  \n"
            
            # print summary in REPL
            out = @capture_out compare_plans(p, r; summary=true)
            @test out == "Range  left: 0U:10U\nRange right: 7U:15U\nVariables only in left plan: [:b]\nVariables only in right plan: [:beta]\n4 common variables.\n:bs differs between the plans for the range(s) 7U:8U.\n"
            
            # return MVTSeries
            out_mvts = compare_plans(p, r)
            @test out_mvts == MVTSeries(7U, (:a,:c,:as,:bs), [0 0 0 0 ; 0 0 0 0 ; 3 3 3 3 ; 2 2 3 3]')

            # return MVTSeries
            out_mvts2 = compare_plans(r, p)
            @test out_mvts2 == MVTSeries(7U, (:a,:c,:as,:bs), [0 0 0 0 ; 0 0 0 0 ; 3 3 3 3 ; 1 1 3 3]')

        end
        # copyto!
        begin
            # When the plans are similar the copy becomes a copy of the source plan
            p_copy = deepcopy(p)
            copyto!(p_copy, q; verbose=false)
            @test p_copy == q

            # When the plans partially overlap, change the overlapping varshks/ranges
            p_copy = deepcopy(p)
            copyto!(p_copy, r; verbose=false)
            @test p_copy.range == p.range
            @test p_copy.varshks == p.varshks
            @test p_copy.exogenous == BitArray([ 
                0  0  0  1  1;
                0  0  0  1  1;
                1  1  0  0  0;
                1  1  0  0  0;
                0  0  0  1  1;
                0  1  0  1  0;
                0  1  0  1  0;
                0  1  0  1  1;
                0  1  0  1  1;
                0  0  0  1  1;
                0  0  0  1  1;
            ])

            # When plans don't overlap, no changes are made
            r2 = Plan(m1, 12U:15U)
            p_copy = deepcopy(p)
            copyto!(p_copy, r2; verbose=false)
            @test p_copy == p

            # when providing a range outside of the plans, an error is thrown
            p_copy = deepcopy(p)
            @test_throws BoundsError copyto!(p_copy, 22U:30U, r; verbose=false)

            # Test warnings
            p_copy = deepcopy(p)
            @test_logs (:warn, ":b is not in the source plan. Skipping.") (
                :warn, "The plan for the following variables/shocks were not copied as they are not in the destination plan:\n [:b]") copyto!(p_copy, r)
        end
    end
end



include("simdatatests.jl")
include("sstests.jl")

@testset "misc" begin
    m = getE3()
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
    as = array2data(a, m.allvars, first(p.range), copy=true)
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

include("misc.jl")

@testset "sparse" begin
    @test (use_pardiso(); StateSpaceEcon.StackedTimeSolver.sf_default == :pardiso)
    @test (use_umfpack(); StateSpaceEcon.StackedTimeSolver.sf_default == :umfpack)
    let m = Model()
        @test (use_pardiso!(m); m.factorization == :pardiso)
        @test (use_umfpack!(m); m.factorization == :umfpack)
    end
end

# make sure Pardiso runs deterministic (single thread) for the tests
Pardiso.set_nprocs_mkl!(1)

for sfdef = QuoteNode.(StateSpaceEcon.StackedTimeSolver.sf_libs)

    sfdef.value == :default && continue

    # Pardiso in macos is giving "Not enough memory".  Disable for now
    # sfdef.value == :pardiso && Sys.isapple() && continue

    @info "Using $(sfdef)"

    Core.eval(StateSpaceEcon.StackedTimeSolver, :(sf_default = $(sfdef)))

    include("simtests.jl")
    include("logsimtests.jl")
    include("sim_fo.jl")
    include("shockdecomp.jl")
    include("dynss.jl")

    include("stochsims.jl")

end

include("modelchanges.jl")