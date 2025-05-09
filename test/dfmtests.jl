##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

using UnPack
using LinearAlgebra
using Statistics

using ModelBaseEcon.DFMModels
using StateSpaceEcon.DFMSolver
using StateSpaceEcon.DFMSolver.Random

@using_example DFM1
@using_example DFM2

if !isdefined(@__MODULE__, :dfm_check_steadystatedata)
function dfm_check_steadystatedata(dfm, data, has_rand_shks)
    @unpack model, params = dfm
    for v in observed(model)
        @test all(data[v] .== params.observed.mean[v])
    end
    for v in states(model)
        @test iszero(data[v])
    end
    for v in shocks(model)
        @test iszero(data[v]) != has_rand_shks
    end
end
end

@testset "ShocksSampler" begin
    # test constructors of various kinds
    @test (ss = ShocksSampler((:a, :b), [3, 4]); ss isa ShocksSampler && ss.names == [:a,:b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler([:a, :b], 3:4); ss isa ShocksSampler && ss.names == [:a,:b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler(["a", "b"], 3:4); ss isa ShocksSampler && ss.names == [:a,:b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler([:a, :b], [3 0; 0 4]); ss isa ShocksSampler && ss.names == [:a,:b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler(["a", "b"], [3 2e-8; 0 4]); ss isa ShocksSampler && ss.names == [:a,:b] && ss.cov == [3.0 1e-8; 1e-8 4.0])
    
    ss = ShocksSampler((:a, :b), [0.09, 0.04])
    @test length(ss) == 2
    nsamples = 10_000_000
    @test (X = rand(ss, nsamples); X isa AbstractMatrix && size(X) == (length(ss),nsamples))
    X = rand(ss, nsamples)
    @test norm(X*X'/nsamples - ss.cov, Inf) < 1e-4
end


@testset "DFM1" begin
    dfm = DFM1.newmodel()
    @unpack model, params = dfm
    @test !any(isnan, params) && lags(dfm) == 1

    rng = 1U:300U
    @test (Plan(dfm, rng); true)
    pl = Plan(dfm, rng)
    @test (rangeof(pl) == 0U:300U)
    
    @test (steadystatedata(dfm, pl) isa MVTSeries)
    @test (steadystatearray(dfm, pl) isa Matrix)
    @test (steadystateworkspace(dfm, pl) isa Workspace)

    @test (z = zerodata(dfm, pl); z isa MVTSeries && iszero(z))
    @test (z = zeroarray(dfm, pl); z isa Matrix && iszero(z))
    @test (z = zeroworkspace(dfm, pl); z isa Workspace && all(iszero, values(z)))

    data = steadystatedata(dfm, pl)
    dfm_check_steadystatedata(dfm, data, false)
    @test (rand_shocks!(dfm, pl, data); true)
    dfm_check_steadystatedata(dfm, data, true)

    data = steadystatedata(dfm, pl)
    ss = ShocksSampler(dfm)
    rand_shocks!(ss, 1U:10U, data)
    dfm_check_steadystatedata(dfm, data, true)
    @test iszero(data[11U:end, shocks(dfm)])

    @test (simulate(dfm, pl, data); true)
    sim = simulate(dfm, pl, data)

end


@testset "DFM2" begin
    dfm = DFM2.newmodel()
    @unpack model, params = dfm
    @test !any(isnan, params) && lags(dfm) == 2

    rng = 2U:300U
    @test (Plan(dfm, rng); true)
    pl = Plan(dfm, rng)
    @test (rangeof(pl) == 0U:300U)
    
    @test (steadystatedata(dfm, pl) isa MVTSeries)
    @test (steadystatearray(dfm, pl) isa Matrix)
    @test (steadystateworkspace(dfm, pl) isa Workspace)

    @test (z = zerodata(dfm, pl); z isa MVTSeries && iszero(z))
    @test (z = zeroarray(dfm, pl); z isa Matrix && iszero(z))
    @test (z = zeroworkspace(dfm, pl); z isa Workspace && all(iszero, values(z)))

    data = steadystatedata(dfm, pl)
    dfm_check_steadystatedata(dfm, data, false)
    @test (rand_shocks!(dfm, pl, data); true)
    dfm_check_steadystatedata(dfm, data, true)

    data = steadystatedata(dfm, pl)
    ss = ShocksSampler(dfm)
    rand_shocks!(ss, 2U:10U, data)
    dfm_check_steadystatedata(dfm, data, true)
    @test iszero(data[11U:end, shocks(dfm)])

    @test (simulate(dfm, pl, data); true)
    sim = simulate(dfm, pl, data)

end




