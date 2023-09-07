##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################


@testset "E1.sstate" begin
    let m = getE1()
        empty!(m.sstate.constraints)
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [false, false, true, true]
        #
        @steadystate m y = 1.2
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [true, false, true, true]
        @test m.sstate.values[1] == 1.2
        #
        m.α = 0.4
        m.β = 1.0 - m.α
        clear_sstate!(m)
        @test m.sstate.mask == trues(4)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(4)
        @test m.sstate.values == [1.2, 0.0, 0.0, 0.0]
        #
        empty!(m.sstate.constraints)
        m.α = 0.3
        m.β = 1.0 - m.α
        clear_sstate!(m)
        @suppress begin
            sssolve!(m; verbose=true, nropts=Options(linesearch=true))
        end
        @test check_sstate(m) == 0
        @test m.sstate.values[2] == 0.0
        #
        empty!(m.sstate.constraints)
        m.α = 0.3
        m.β = 1.0 - m.α
        clear_sstate!(m)
        @test_throws ErrorException sssolve!(m, method=:other)
    end
end

@testset "E1.sstate, auto" begin
    let m = getE1()
        empty!(m.sstate.constraints)
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m; method=:auto)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [true, true, true, true]  # different from default
        #
        @steadystate m y = 1.2
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m; method=:auto)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [true, true, true, true] # different from default
        @test m.sstate.values[1] == 1.2
        #
        m.α = 0.4
        m.β = 1.0 - m.α
        clear_sstate!(m)
        @test m.sstate.mask == trues(4)
        sssolve!(m; method=:auto)
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(4)
        @test m.sstate.values == [1.2, 0.0, 0.0, 0.0]
        #
        empty!(m.sstate.constraints)
        m.α = 0.3
        m.β = 1.0 - m.α
        clear_sstate!(m)
        @suppress begin
            sssolve!(m; method=:auto, verbose=true)
        end
        @test check_sstate(m) == 0
        @test m.sstate.values[2] == 0.0
    end
end

@testset "E1.sstate, lm" begin
    let m = getE1()
        empty!(m.sstate.constraints)
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m; method=:lm)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [true, true, true, true]  # different from default
        #
        @steadystate m y = 1.2
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m; method=:lm)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [true, true, true, true]  # different from default
        @test m.sstate.values[1] == 1.2
        #
        m.α = 0.4
        m.β = 1.0 - m.α
        clear_sstate!(m)
        @test m.sstate.mask == trues(4)
        sssolve!(m; method=:lm)
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(4)
        @test m.sstate.values == [1.2, 0.0, 0.0, 0.0]
        #
        empty!(m.sstate.constraints)
        m.α = 0.3
        m.β = 1.0 - m.α
        clear_sstate!(m)
        @suppress begin
            sssolve!(m; method=:lm, verbose=true)
        end
        @test check_sstate(m) == 0
        @test m.sstate.values[2] == 0.0
    end
end

@testset "E2.sstate" begin
    let m = getE2()
        empty!(m.sstate.constraints)
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(12)
        @test m.sstate.values == zeros(12)

        empty!(m.sstate.constraints)
        @steadystate m rate = 0.0
        clear_sstate!(m)
        @test m.sstate.mask[3] == true
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(12)
        @test m.sstate.values == zeros(12)

        empty!(m.sstate.constraints)
        @steadystate m rate = 0.1
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) > 0
    end
end

@testset "E6.sstate" begin
    let m = getE6()
        m.options.maxiter = 50

        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test sum(m.sstate.mask) == 9 + 4
        @test m.sstate.values[m.sstate.mask] ≈ [0.005, 0.0, 0.0045, 0.0, 0.0095, 0.0, 0.005, 0.0045, 0.0095, 0, 0, 0, 0]

        empty!(m.sstate.constraints)
        @steadystate m lp = 2
        @steadystate m ly = 3
        @steadystate m lyn = 7
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test all(m.sstate.mask)
        @test m.sstate.values ≈ [0.005, 0.0, 0.0045, 0.0, 0.0095, 0.0, 2.0, 0.005, 3.0, 0.0045, 7.0, 0.0095, 0, 0, 0, 0]
    end
end

@testset "E7.sstate" begin
    let m = getE7()
        m.options.maxiter = 100
        m.options.tol = 1e-9

        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        # underdetermined system - two free variables
        @test sum(m.sstate.mask) == length(m.sstate.mask) - 2

        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m, tol=10m.tol) == 0
        @test all(m.sstate.mask)
        @test m.sstate.values ≈ [0.004, 0.0, 0.004, 0.0, 0.004, 0.0, 14.0, 0.004, 7.0, 0.004, 9.267287357063445, 0.004, 14.000911466453774, 0.004, 0, 0, 0, 0, 14.000911466453774, 0.004, 9.267287357063445, 0.004]

        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        @steadystate m linv = 8
        clear_sstate!(m)
        sssolve!(m)
        # overdetermined inconsistent set of constraints
        @test check_sstate(m) > 0

        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        @steadystate m ly = log(exp(lc) + exp(linv))
        clear_sstate!(m)
        sssolve!(m)
        # overdetermined consistent set of constraints
        @test check_sstate(m) == 0
        @test all(m.sstate.mask)
        @test m.sstate.values ≈ [0.004, 0.0, 0.004, 0.0, 0.004, 0.0, 14.0, 0.004, 7.0, 0.004, 9.267287357063445, 0.004, 14.000911466453774, 0.004, 0, 0, 0, 0, 14.000911466453774, 0.004, 9.267287357063445, 0.004]
    end
end

##

using ModelBaseEcon
using TimeSeriesEcon
using StateSpaceEcon

module SSTEST
using ModelBaseEcon
model = Model()
@steadyvariables model a b
@variables model c
@equations model begin
    a[t] = 0.95b[t] + 0.1
    b[t+1] = 0.1b[t] + c[t]
    c[t] = 1.2
end
@initialize model
end

@testset "SSTEST" begin
    let m = deepcopy(SSTEST.model)
        @test sum(issteady.(m.allvars)) == 2
        clear_sstate!(m)
        @test issssolved(m)
        @test check_sstate(m) == 0
    end
end

module DiagnoseSState
using ModelBaseEcon
model = Model()
@steadyvariables model a b
@variables model c
@equations model begin
    a[t] = (a[t-1] + 10)^2
    b[t] = b[t-1] + 10
    b[t] = (a[t-1] + 10)^2
    c[t] = a[t] + b[t]
end
@initialize model
end

@testset "DiagnoseSState" begin
    let m = Model()
        @steadyvariables m a b
        @variables m c
        @equations m begin
            a[t] = (a[t-1] + 10)^2
            b[t] = b[t-1] + 10
            b[t] = (a[t-1] + 10)^2
            c[t] = a[t] + b[t]
        end
        @initialize m

        # tests
        @test sum(issteady.(m.allvars)) == 2
        clear_sstate!(m)
        @test issssolved(m) == false
        @test check_sstate(m) == 8
        out = @capture_err begin
            sssolve!(m; verbose=true)
            @test check_sstate(m; verbose=true) == 6
        end
        @test occursin("System may be inconsistent", out)

    end
end

@testset "Inadmissible" begin
    let m = Model()
        m.flags.ssZeroSlope = true
        @steadyvariables m a b
        @variables m c
        @equations m begin
            a[t] = (a[t-1] + 10)^2
            b[t] = b[t-1] / 0
            b[t] = (a[t-1] + 10)^2
            c[t] = a[t] + b[t]
        end
        @initialize m

        # tests
        @test sum(issteady.(m.allvars)) == 2
        clear_sstate!(m)
        @test issssolved(m) == false
        out = @capture_err begin
            @test_throws ErrorException check_sstate(m) == 8
        end
        @test occursin("Inadmissible point in steady state equation", out)
        out = @capture_err begin
            @test_throws ErrorException sssolve!(m; presolve=true)
        end
        @test occursin("Inadmissible point in steady state equation", out)
    end
end

@testset "presolve, ssZeroSlope" begin
    let m = getE1()
        empty!(m.sstate.constraints)
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m; presolve=false)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [false, false, true, true]
    end
    
    let m = getE1()
        empty!(m.sstate.constraints)
        m.α = 0.5
        m.β = 1.0 - m.α
        m.flags.ssZeroSlope = true
        clear_sstate!(m)
        sssolve!(m; presolve=false)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [false, true, true, true]
    end
end