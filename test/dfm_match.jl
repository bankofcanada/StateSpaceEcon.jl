##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

##################################################################################
# DFM EM solver legacy-match tests.
#
# Builds the four legacy example models (DFM1/DFM2/DFM3/DFM3MQ) via the
# functional DSL (a verbatim transcription of ModelBaseEcon.jl/examples/DFM*.jl -
# the DSL is a verbatim port, so the builders are identical), loads the legacy EM
# initial param sets, runs `EMestimate!`, and asserts the final parameter
# vector AND the smoothed factor estimates match legacy at atol=1e-8.
# Reference is legacy_reference/dfm_reference.json
# (capture_dfm.jl); the test reads ONLY the JSON.
#
# Coverage: EM full-data match (DFM1, DFM2/3, DFM3MQ);
# missing-data (Banbura-Modugno) EM match; filter/smoother matrix-level match;
# ShocksSampler; em_apply_constraint!; em_impute_{kalman,interpolation}!.
# The legacy TimeSeriesEcon-typed Plan/steadystatedata/simulate/rand_shocks!
# convenience surface is a documented scoped-defer (see src/dfm.jl).
#
# Per-cell-collapse convention: each param/factor matrix collapses to
# one `max(abs(got - want))` assertion.
##################################################################################

using Test
using StateSpaceEcon
using StateSpaceEcon: DFMSolver
using StateSpaceEcon.DFMSolver: ShocksSampler
using ModelBaseEcon
using ModelBaseEcon.DFMModels
using LinearAlgebra
using JSON

const _DFM_REF_PATH = joinpath(@__DIR__, "..", "..", "legacy_reference", "dfm_reference.json")

# JSON null (NaN sentinel) -> NaN; numbers -> Float64
_dfm_num(x) = x === nothing ? NaN : Float64(x)
_dfm_vec(a) = Float64[_dfm_num(x) for x in a]
_dfm_mat(a, r, c) = reshape(_dfm_vec(a), r, c)

# ---- model builders (verbatim transcription of the legacy examples) ----------

function build_dfm1()
    m = DFM(:example_dfm1)
    add_observed!(m, :a, :b)
    add_components!(m, F=CommonComponents("F"))
    map_loadings!(m, (:a, :b) => :F)
    add_shocks!(m, :a, :b)
    initialize_dfm!(m)
    return m
end

function build_dfm2()
    m = DFM(:example_dfm2)
    add_components!(m,
        F=CommonComponents("F", order=2),
        G=CommonComponents("G", order=2),
        ic=IdiosyncraticComponents(),
    )
    map_loadings!(m,
        (:a, :b) => :F,
        (:a, :c, :d) => :G,
        (:c, :d) => :ic,
    )
    add_shocks!(m, :a, :b, :c, :d)
    initialize_dfm!(m)
    return m
end

function build_dfm3()
    m = DFM(:example_dfm3)
    M_VARS = (:a, :b, :c)
    Q_VARS = (:y, :z)
    D_VAR = :k
    add_observed!(m,
        :obsM => ObservedBlock(M_VARS),
        :obsQ => ObservedBlock(Q_VARS),
        :obsQ => :k,
    )
    add_components!(m,
        F=CommonComponents((:U, :G), order=2),
        corM=IdiosyncraticComponents(),
        corQ=IdiosyncraticComponents(),
    )
    map_loadings!(m,
        (:a, :b, :y, :k) => :U,
        (:c, :z, :k) => :G,
        M_VARS => :corM,
        Q_VARS => :corQ,
        :k => :corQ,
    )
    add_shocks!(m, M_VARS, Q_VARS, D_VAR)
    initialize_dfm!(m)
    return m
end

function build_dfm3mq()
    m = DFM(:example_dfm3mq)
    M_VARS = (:a, :b, :c)
    Q_VARS = (:y, :z)
    D_VAR = :k
    add_observed!(m,
        :obsM => ObservedBlock(M_VARS),
        :obsQ => ObservedBlock(MixFreq{:MQ}, Q_VARS),
        :obsQ => :k,
    )
    add_components!(m,
        F=CommonComponents(MixFreq{:MQ}, (:U, :G), order=2),
        corM=IdiosyncraticComponents(),
        corQ=IdiosyncraticComponents(MixFreq{:MQ}),
    )
    map_loadings!(m,
        (:a, :b, :y, :k) => :U,
        (:c, :z, :k) => :G,
        M_VARS => :corM,
        Q_VARS => :corQ,
        :k => :corQ,
    )
    add_shocks!(m, M_VARS, Q_VARS, D_VAR)
    initialize_dfm!(m)
    return m
end

# ---- factor-estimate extraction (smoother x_smooth at converged params) ------

function _dfm_factor_estimates(m::DFM, Y::AbstractMatrix)
    LM = DFMSolver.kf_linear_model(m)
    kfd = StateSpaceEcon.Kalman.KFDataSmoother(Float64, size(Y, 1), m, Y)
    kf = StateSpaceEcon.Kalman.KFilter(kfd)
    nx = DFMSolver.kf_length_x(m)
    x0 = zeros(nx)
    Px0 = Matrix{Float64}(1e-10 * I(nx))
    anymissing = any(isnan, Y)
    StateSpaceEcon.Kalman.kf_filter!(kf, Y, x0, Px0, LM; fwdstate=false, anymissing)
    StateSpaceEcon.Kalman.kf_smoother!(kf, LM; fwdstate=false)
    return kfd.x_smooth   # NS_with_lags x NT
end

# ---- per-model EM legacy-match (params + factor estimates) -------------------

function _run_dfm_em_match(modname::String, builder, ref; kwargs...)
    d = ref[modname]
    NT = Int(d["NT"]); NO = Int(d["NO"])
    Y0 = _dfm_mat(d["sim_obs"], NT, NO)

    inits = d["inits"]
    recomputed = d["recomputed"]
    finals = d["finals"]

    @testset "$modname EM" begin
        for i_str in sort(collect(keys(inits)); by=x -> parse(Int, x))
            init = _dfm_vec(inits[i_str])
            want_final = _dfm_vec(finals[i_str])
            rec = recomputed[i_str]
            want_recomp = _dfm_vec(rec["final"])
            want_xsm = _dfm_mat(rec["x_smooth"], Int(rec["x_smooth_rows"]), Int(rec["x_smooth_cols"]))

            m = builder()
            copyto!(m.params, init)
            DFMSolver.EMestimate!(m, copy(Y0); verbose=false, strict=false, kwargs...)
            got_final = Float64[Float64(v) for v in m.params]

            # params vs the daec-stored final
            dmax = maximum(abs, got_final .- want_final)
            @test dmax ≤ 1e-8

            # params vs the legacy-recomputed final (current legacy code)
            dmax_rec = maximum(abs, got_final .- want_recomp)
            @test dmax_rec ≤ 1e-8

            # factor estimates vs legacy smoother
            got_xsm = _dfm_factor_estimates(m, copy(Y0))
            fmax = maximum(abs, got_xsm .- want_xsm)
            @test fmax ≤ 1e-8
        end
    end
end

# ---- missing-data EM legacy-match (Banbura & Modugno path) -------------------
# Y with the legacy `miss` mask applied (-> NaN). EM treats missing data per
# Banbura & Modugno 2014 (impute_missing=false default). Matches em_miss_p{i}.

function _run_dfm_em_miss_match(modname::String, builder, ref; kwargs...)
    d = ref[modname]
    haskey(d, "miss") || return
    haskey(d, "miss_finals") && !isempty(d["miss_finals"]) || return
    NT = Int(d["NT"]); NO = Int(d["NO"])
    Y0 = _dfm_mat(d["sim_obs"], NT, NO)
    miss = _dfm_mat(d["miss"], NT, NO) .!= 0.0

    inits = d["inits"]
    miss_finals = d["miss_finals"]

    @testset "$modname EM missing" begin
        for i_str in sort(collect(keys(miss_finals)); by=x -> parse(Int, x))
            haskey(inits, i_str) || continue
            init = _dfm_vec(inits[i_str])
            want = _dfm_vec(miss_finals[i_str])

            Y = copy(Y0)
            Y[miss] .= NaN
            m = builder()
            copyto!(m.params, init)
            DFMSolver.EMestimate!(m, Y; verbose=false, strict=false, kwargs...)
            got = Float64[Float64(v) for v in m.params]
            dmax = maximum(abs, got .- want)
            @test dmax ≤ 1e-8
        end
    end
end

# ---- filter / smoother matrix-level legacy-match -----------------------------
# Reproduces the legacy do_test_filter numerical match without the TimeSeriesEcon
# Plan/MVTSeries machinery: build the DFM at default params, run kf_filter /
# kf_smoother on the captured filter input Y (x0=0, Px0=I), and compare the
# filter (x, x_pred, y_pred) and smoother (x_smooth, y_smooth) outputs to the
# captured legacy outputs at atol=1e-8.

function _run_dfm_filter_match(modname::String, builder, ref)
    d = ref[modname]
    haskey(d, "filter") || return
    f = d["filter"]
    Y = _dfm_mat(f["Y"], Int(f["Y_rows"]), Int(f["Y_cols"]))

    m = builder()
    copyto!(m.params, _dfm_vec(d["default_params"]))
    nx = nstates_with_lags(m)
    x0 = zeros(nx)
    Px0 = Matrix{Float64}(I(nx))

    kfd_f = StateSpaceEcon.Kalman.kf_filter(Y, x0, Px0, m)
    kfd_s = StateSpaceEcon.Kalman.kf_smoother(Y, x0, Px0, m)

    @testset "$modname filter/smoother" begin
        @test maximum(abs, kfd_f.x .- _dfm_mat(f["x"], Int(f["x_rows"]), Int(f["x_cols"]))) ≤ 1e-8
        @test maximum(abs, kfd_f.x_pred .- _dfm_mat(f["x_pred"], Int(f["x_pred_rows"]), Int(f["x_pred_cols"]))) ≤ 1e-8
        @test maximum(abs, kfd_f.y_pred .- _dfm_mat(f["y_pred"], Int(f["y_pred_rows"]), Int(f["y_pred_cols"]))) ≤ 1e-8
        @test maximum(abs, kfd_s.x_smooth .- _dfm_mat(f["x_smooth"], Int(f["x_smooth_rows"]), Int(f["x_smooth_cols"]))) ≤ 1e-8
        @test maximum(abs, kfd_s.y_smooth .- _dfm_mat(f["y_smooth"], Int(f["y_smooth_rows"]), Int(f["y_smooth_cols"]))) ≤ 1e-8
    end
end

# ---- ShocksSampler - Distributions-based, no TimeSeriesEcon ------------------

function _run_shocks_sampler_tests()
    @testset "ShocksSampler" begin
        ss = ShocksSampler((:a, :b), [3, 4])
        @test ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0]
        ss = ShocksSampler([:a, :b], 3:4)
        @test ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0]
        ss = ShocksSampler(["a", "b"], 3:4)
        @test ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0]
        ss = ShocksSampler([:a, :b], [3 0; 0 4])
        @test ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 0; 0 4.0]
        ss = ShocksSampler(["a", "b"], [3 2e-8; 0 4])
        @test ss isa ShocksSampler && ss.names == [:a, :b] && ss.cov == [3.0 1e-8; 1e-8 4.0]

        ss = ShocksSampler((:a, :b), [0.09, 0.04])
        @test length(ss) == 2
        nsamples = 1_000_000
        X = rand(ss, nsamples)
        @test X isa AbstractMatrix && size(X) == (length(ss), nsamples)
        # covariance recovery (diagonal sampler)
        @test norm(X * X' / nsamples - ss.cov, Inf) < 1e-2

        # build a ShocksSampler from a DFM and draw -- symmetric/cholesky path
        m = build_dfm1()
        copyto!(m.params, ones(length(m.params)))
        ssd = ShocksSampler(m)
        @test ssd isa ShocksSampler && length(ssd) == nshocks(m)
        @test rand(ssd, 5) isa AbstractMatrix

        io = IOBuffer()
        show(io, ss)
        s = String(take!(io))
        @test occursin("ShocksSampler", s) && occursin("shocks:", s) && occursin("covariance:", s)
    end
end

# ---- em_apply_constraint! - matrix-level ------------------------------------

function _run_em_constraint_tests()
    @testset "EM matrix constraint" begin
        A = rand(3, 2)
        W = zeros(2, 6)
        q = zeros(2)
        # force A[2,2] = 3
        W[1, 5] = 1
        q[1] = 3
        # force 5A[3,1] = A[3,2] + 1   ->  5*A[3,1] - A[3,2] = 1
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
        @test maximum(abs, A[same] .- B[same]) < 1e-10
        @test abs(B[2, 2] - q[1]) < 1e-10
        @test abs(5B[3, 1] - (B[3, 2] + q[2])) < 1e-10
    end
end

# ---- impute helpers - matrix-level ------------------------------------------

function _run_impute_tests(ref)
    @testset "DFM impute helpers" begin
        d = ref["dfm2"]
        f = d["filter"]
        Y = _dfm_mat(f["Y"], Int(f["Y_rows"]), Int(f["Y_cols"]))
        m = build_dfm2()
        copyto!(m.params, _dfm_vec(d["default_params"]))
        nx = nstates_with_lags(m)
        kfd = StateSpaceEcon.Kalman.kf_smoother(Y, zeros(nx), Matrix{Float64}(I(nx)), m)

        # no-op when EY === Y
        @test DFMSolver.em_impute_kalman!(Y, Y, kfd) === Y
        @test DFMSolver.em_impute_interpolation!(Y, Y) === Y

        # punch 20% holes deterministically
        Ymiss = copy(Y)
        miss = falses(size(Y))
        for i in 1:5:length(Y)
            miss[i] = true
        end
        Ymiss[miss] .= NaN

        EY = copy(Ymiss)
        DFMSolver.em_impute_kalman!(EY, Ymiss, kfd)
        # non-missing entries unchanged; missing entries filled from smoother y_smooth
        @test all(EY[.!miss] .== Ymiss[.!miss])
        @test !any(isnan, EY)

        EY2 = copy(Ymiss)
        DFMSolver.em_impute_interpolation!(EY2, Ymiss)
        @test all(EY2[.!miss] .== Ymiss[.!miss])
        @test !any(isnan, EY2)
    end
end

function run_dfm_match_tests!()
    # The numerical legacy-match needs the committed reference data; it
    # self-skips when that file is absent. The structural testsets below
    # (sampler, constraint, impute) build their own inputs and always run.
    have_ref = isfile(_DFM_REF_PATH)
    ref = have_ref ? JSON.parsefile(_DFM_REF_PATH) : nothing

    @testset "DFM EM legacy-match" begin
        if !have_ref
            @info "skipping DFM EM legacy-match - reference data not present"
            @test_skip "dfm_reference.json not present"
        else
            _run_dfm_em_match("dfm1", build_dfm1, ref)
            _run_dfm_em_match("dfm2", build_dfm2, ref; rftol=2e-5)
            _run_dfm_em_match("dfm3", build_dfm3, ref; rftol=1e-5, maxiter=1000)
            _run_dfm_em_match("dfm3mq", build_dfm3mq, ref; rftol=1e-5, maxiter=1000)

            # missing-data (Banbura & Modugno) EM path
            _run_dfm_em_miss_match("dfm1", build_dfm1, ref)
            _run_dfm_em_miss_match("dfm2", build_dfm2, ref; rftol=2e-5)
            _run_dfm_em_miss_match("dfm3", build_dfm3, ref; rftol=1e-5, maxiter=1000)
            _run_dfm_em_miss_match("dfm3mq", build_dfm3mq, ref; rftol=1e-5, maxiter=1000)

            # filter/smoother matrix-level match (all four models)
            _run_dfm_filter_match("dfm1", build_dfm1, ref)
            _run_dfm_filter_match("dfm2", build_dfm2, ref)
            _run_dfm_filter_match("dfm3", build_dfm3, ref)
            _run_dfm_filter_match("dfm3mq", build_dfm3mq, ref)

            _run_impute_tests(ref)
        end

        # Structural testsets - no reference data needed.
        _run_shocks_sampler_tests()
        _run_em_constraint_tests()
    end
end
