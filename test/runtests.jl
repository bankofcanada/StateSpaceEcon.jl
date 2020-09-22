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

using Test
using Suppressor

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
    m = E1.model
    p = Plan(m, 1:3)
    @test first(p.range) == ii(0)
    @test last(p.range) == ii(4)
    out = @capture_out print(p)
    @test length(split(out, '\n')) == 2
    @test p[1] == [:y_shk]
    @test p[ii(1)] == [:y_shk]
    endogenize!(p, :y_shk, ii(1))
    @test isempty(p[ii(1)])
    endogenize!(p, :y_shk, ii(1):ii(3))
    exogenize!(p, :y, ii(2))
    exogenize!(p, :y, ii(4))
    # make sure indexing with integers works as well
    @test p[ii(0)] == p[1] == [:y_shk]
    @test p[ii(1)] == p[2] == []
    @test p[ii(2)] == p[3] == [:y]
    @test p[ii(3)] == p[4] == []
    @test p[ii(4)] == p[5] == [:y, :y_shk]
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
end

include("simdatatests.jl")
include("sstests.jl")

using StateSpaceEcon.StackedTimeSolver: dict2array, dict2data, array2dict, array2data, data2dict, data2array
@testset "dict2array" begin
    m = E3.model
    sim = m.maxlag .+ (1:10)
    p = Plan(m, sim)

    # random data
    d = zerodict(m, p)
    for v in values(d)
        v .= rand(Float64, size(v))
    end

    @test size(dict2array(d, [:pinf, :ygap])) == (length(p.range), 2)
    @test size(dict2array(d, ["pinf", "ygap"])) == (length(p.range), 2)
    @test size(dict2array(d, m.variables)) == (length(p.range), 3)
    a = dict2array(d, m.allvars)
    @test size(a) == (length(p.range), 6)
    @test a == hcat((d[string(v)] for v in m.allvars)...)

    # error variable missing from dictionary
    @test_throws ArgumentError dict2array(d, [:pinf, :ygap, :nosuchvar])
    # error out of range
    @test_throws ArgumentError dict2array(d, [:pinf, :ygap], range=10U:20U)

    # warning variables with different ranges
    d["wrong_var"] = TSeries(3U, rand(10))
    b = @test_logs (:warn, r".*Using\s+intersection\s+range: 3U:12U") dict2array(d, [:pinf, :wrong_var])
    @test size(b) == (10, 2)
    @test b[:,1] == d["pinf"][3U:12U].values
    @test b[:,2] == d["wrong_var"][3U:12U].values

    s = dict2data(d, m.allvars)
    @test s == a

    sa = data2array(s)
    @test sa == s
    s.pinf[3U:5U] = 3:5
    @test sa == s

    sd = data2dict(s)
    @test Set(keys(sd)) == Set(string.(colnames(s)))

    as = array2data(a, m.allvars, first(p.range))
    @test as == a
    a[1,2] = 2.5
    @test as == a
    as = array2data(a, m.allvars, first(p.range), copy=true)
    @test as == a
    a[1,2] = 3.0
    @test as != a
    as[1,2] = 3.0
    @test as == a

    ad = array2dict(a, m.allvars, first(p.range))
    @test length(ad) == size(a, 2)
    @test all(ad[string(v)].values == a[:,i] for (i, v) in enumerate(m.allvars))
end

include("simtests.jl")
