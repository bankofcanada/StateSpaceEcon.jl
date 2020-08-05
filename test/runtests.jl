using StateSpaceEcon
using TimeSeriesEcon
using ModelBaseEcon

@info "Loading Model examples. Might take some time to pre-compile."
@using_example M1
@using_example M2
@using_example M3
@using_example M3nl
@using_example M6
@using_example M7
@using_example M7A

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
    m = M1.model
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
end

include("simdatatests.jl")
include("sstests.jl")
include("simtests.jl")
