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

using TimeSeriesEcon.DataEcon

@using_example DFM1
@using_example DFM2
@using_example DFM3
@using_example DFM3MQ

if !isdefined(@__MODULE__, :dfm_check_steadystatedata)
    function dfm_check_steadystatedata(dfm, data, has_rand_shks)
        @unpack model, params = dfm
        @test (size(data,2) == nvarshks(dfm))
        c = 0
        for on in keys(model.observed)
            pp = params[on].mean
            for var in keys(pp)
                @test all(data[var] .== pp[var])
                c = c + 1
            end
        end
        for v in states(model)
            @test iszero(data[v])
            c = c + 1
        end
        for v in shocks(model)
            @test iszero(data[v]) != has_rand_shks
            c = c + 1
        end
        @test c == size(data,2)
    end
end

##

@testset "ShocksSampler" begin
    # test constructors of various kinds
    @test (ss = ShocksSampler((:a, :b), [3, 4]); ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler([:a, :b], 3:4); ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler(["a", "b"], 3:4); ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler([:a, :b], [3 0; 0 4]); ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0])
    @test (ss = ShocksSampler(["a", "b"], [3 2e-8; 0 4]); ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 1e-8; 1e-8 4.0])

    ss = ShocksSampler((:a, :b), [0.09, 0.04])
    @test length(ss) == 2
    nsamples = 10_000_000
    @test (X = rand(ss, nsamples); X isa AbstractMatrix && size(X) == (length(ss), nsamples))
    X = rand(ss, nsamples)
    @test norm(X * X' / nsamples - ss.cov, Inf) < 1e-4

    @test begin
        search_output = occursin(let io = IOBuffer()
            show(io, ss)
            String(take!(io))
        end)
        search_output("ShocksSampler") & search_output("shocks:") && search_output("covariance:")
    end
end


if !isdefined(@__MODULE__, :do_test_dfm)
    function do_test_dfm(MOD, nlags)
        dfm = MOD.newmodel()
        @unpack model, params = dfm
        @test !any(isnan, params) && lags(dfm) == nlags
    
        rng = 1U:300U
        @test (Plan(dfm, rng); true)
        pl = Plan(dfm, rng)
        @test (rangeof(pl) == (1-nlags)*U:300U)
        vars = [i for (i, v) in enumerate(varshks(dfm)) if v ∉ shocks(dfm)]
        shks = [i for (i, v) in enumerate(varshks(dfm)) if v ∈ shocks(dfm)]
        @test all(pl.exogenous[:, shks])
        @test !any(pl.exogenous[:, vars])
    
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
    
        exog_endo!(pl, [:a], [:a_shk], rng[1:5])
        @test_throws "Non-empty plan" simulate(dfm, pl, data)
    
    end
end

@testset "DFM" begin
    do_test_dfm(DFM1, 1)
    do_test_dfm(DFM2, 2)
    do_test_dfm(DFM3, 2)
    do_test_dfm(DFM3MQ, 5)
end

##

td = DataEcon.readdb(joinpath(@__DIR__, "data", "dfm.daec"))
if !isdefined(@__MODULE__, :do_test_filter)
    function do_test_filter(MOD, TD)
        dfm = MOD.newmodel()
        rng = rangeof(TD.shks; drop=lags(dfm))
        p = Plan(dfm, rng)
        data = steadystatedata(dfm, p)
        data[rng, :] .= 0
        data[rng, :] = TD.shks
        sim = simulate(dfm, p, data)
        @test @compare sim TD.sim quiet
    
        Y = sim[rng, observed(dfm)].values
        x0 = zeros(nstates_with_lags(dfm))
        Px0 = I(nstates_with_lags(dfm))
        kfd = kf_filter(Y, x0, Px0, dfm)
        @test @compare kfd2data(kfd, :update, dfm, rng) TD.update quiet
        @test @compare kfd2data(kfd, :pred, dfm, rng) TD.pred quiet
        kfd = kf_smoother(Y, x0, Px0, dfm)
        @test @compare kfd2data(kfd, :smooth, dfm, rng) TD.smooth quiet
    
        for which in (:update, :pred, :smooth)
            a1 = kfd2data(kfd, which, dfm, rng; states_with_lags=false)
            a2 = kfd2data(kfd, which, dfm, rng; states_with_lags=true)
            @test @compare(a1, a2, quiet) == (lags(dfm) == 1)
            @test @compare(a1, a2, quiet, ignoremissing) == true
        end
    end
end

@testset "DFM Filter" begin
    do_test_filter(DFM1, td.dfm1)
    do_test_filter(DFM2, td.dfm2)
    do_test_filter(DFM3, td.dfm3)
    do_test_filter(DFM3MQ, td.dfm3mq)
end

##

@testset "DFM2 Missing" begin
    dfm = DFM2.newmodel()
    rng = rangeof(td.dfm2.shks; drop=lags(dfm))
    p = Plan(dfm, rng)
    data = steadystatedata(dfm, p)
    data[rng, :] .= 0
    data[rng, :] = td.dfm2.shks
    sim = simulate(dfm, p, data)
    @test @compare sim td.dfm2.sim quiet

    Y = sim[rng, observed(dfm)].values
    # make 20% missing values in Y 
    miss = rand(length(Y)) .< 0.2
    Y[miss] .= NaN

    # run filter and smoother on data with missing values
    x0 = zeros(nstates_with_lags(dfm))
    Px0 = I(nstates_with_lags(dfm))
    kfd = kf_smoother(Y, x0, Px0, dfm)

    # Don't overwrite missing values pattern in original data
    @test DFMSolver.em_impute_kalman!(Y, Y, kfd) === Y
    @test DFMSolver.em_impute_interpolation!(Y, Y) === Y
    
    # Interpolate by fetching missing values from the smoother results
    EY = copy(Y)
    DFMSolver.em_impute_kalman!(EY, Y, kfd)
    @test compare(EY[.!miss], Y[.!miss], quiet=true)
    SY = kfd2data(kfd, :smooth, dfm, rng; states_with_lags=true)[:, observed(dfm)]
    @test compare(EY[miss], SY[miss], quiet=true)
    
    # Interpolate by cubic interpolation
    DFMSolver.em_impute_interpolation!(EY, Y)
    @test compare(EY[.!miss], Y[.!miss], quiet=true)
    @test compare(EY[miss], SY[miss], quiet=true) == false # interpolation is not the same as 

end

##

@testset "EM matrix constraint" begin

    A = rand(3, 2)
    W = zeros(2, 6)
    q = zeros(2)
    # force A[2,2] = 3
    W[1, 5] = 1
    q[1] = 3
    # force 5A[3,1]  = A[3,2] + 1
    W[2, 3] = 5
    W[2, 6] = -1
    q[2] = 1

    mc = DFMSolver.EM_MatrixConstraint(2, W, q)
    cXTX = cholesky(Matrix{Float64}(I(3)))
    Σ = Matrix{Float64}(I(3))
    
    B = DFMSolver.em_apply_constraint!(copy(A), nothing, cXTX, Σ)
    @test norm(B - A) < 1e-14
    
    same = [true, true, false, true, false, false]
    B = DFMSolver.em_apply_constraint!(B, mc, cXTX, Σ)
    @test compare(A[same], B[same], quiet=true)
    @test compare(B[2,2], q[1], quiet=true)
    @test compare(5B[3,1], B[3,2]+q[2], quiet=true)

end

##

@testset "DFM1 EM" begin
    m = DFM1.newmodel()
    true_p = copy(m.params)
    Y = td.dfm1.sim[1U:end, observed(m)].values

    fill!(m.params, NaN)
    m.params.F.covar .= 1.0
    DFMSolver.EMestimate!(m, Y)
    @test compare(m.params, td.dfm1.em_p1, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.loadings.F[2,1] = 0.9
    m.params.F.covar .= 1.0
    DFMSolver.EMestimate!(m, Y)
    @test compare(m.params, td.dfm1.em_p2, quiet=true)
    
    # -----------------------
    Y[td.dfm1.miss] .= NaN
    
    fill!(m.params, NaN)
    m.params.F.covar .= 1.0
    DFMSolver.EMestimate!(m, Y)
    @test compare(m.params, td.dfm1.em_miss_p1, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.loadings.F[2,1] = 0.9
    m.params.F.covar .= 1.0
    DFMSolver.EMestimate!(m, Y)
    @test compare(m.params, td.dfm1.em_miss_p2, quiet=true)
end

##

@testset "DFM2 EM" begin
    m = DFM2.newmodel()
    true_p = copy(m.params)
    Y = td.dfm2.sim[1U:end, observed(m)].values

    fill!(m.params, NaN)
    m.params.F.covar .= 1.0
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_p1, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.loadings.F[2] = 0.9
    m.params.observed.loadings.G[1] = 1.1
    m.params.F.covar .= 1.0
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_p2, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.loadings.F[1] = -0.2
    m.params.observed.loadings.F[2] = 0.9
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_p3, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.mean = [2.3,-1.5,1.2,0.0]
    m.params.F.covar .= 1.0
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_p4, quiet=true)

    # -----------------------
    Y[td.dfm2.miss] .= NaN
    
    fill!(m.params, NaN)
    m.params.F.covar .= 1.0
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_miss_p1, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.loadings.F[2] = 0.9
    m.params.observed.loadings.G[1] = 1.1
    m.params.F.covar .= 1.0
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_miss_p2, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.loadings.F[2] = 0.9
    m.params.observed.loadings.G[1] = 1.1
    m.params.F.covar .= 1.0
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false, use_full_XTX=false)
    @test compare(m.params, td.dfm2.em_miss_p2f, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.loadings.F[1] = -0.2
    m.params.observed.loadings.F[2] = 0.9
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_miss_p3, quiet=true)
    
    fill!(m.params, NaN)
    m.params.observed.mean = [2.3,-1.5,1.2,0.0]
    m.params.F.covar .= 1.0
    m.params.G.covar .= 1.0
    DFMSolver.EMestimate!(m, Y, rftol=2e-5, verbose=false)
    @test compare(m.params, td.dfm2.em_miss_p4, quiet=true)

end

## 

@testset "DFM3 EM" begin
    m = DFM3.newmodel()
    true_p = copy(m.params)
    Y = td.dfm3.sim[1U:end, observed(m)].values

    fill!(m.params, NaN)
    DFMSolver.EMestimate!(m, Y, rftol=1e-5, verbose=false)
    @test compare(m.params, td.dfm3.em_p1, quiet=true)
    
end
