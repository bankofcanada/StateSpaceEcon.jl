##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Kalman filter / smoother tests.
#
# Verifies the Durbin-Koopman filter + smoother in
# StateSpaceEcon.jl/src/kalman/ against:
#   1. A legacy numerical reference (legacy_reference/kalman_reference.json,
#      captured by legacy_reference/capture_kalman.jl from the legacy
#      StateSpaceEcon.Kalman module) - the done-criterion: the
#      log-likelihood agrees with legacy StateSpaceEcon to floating-point
#      tolerance on a 3-equation linear test model.
#   2. Self-consistency invariants - covariances symmetric PSD, smoother
#      variance <= filter variance, container plumbing.
#
# The reference model is a fixed 3-state / 3-observable time-invariant
# linear state-space model; this file rebuilds it identically.
# ----------------------------------------------------------------------

using JSON
using LinearAlgebra

# Reference model constants - must match legacy_reference/capture_kalman.jl
# verbatim (that script is the source of the committed reference JSON).
const KF_MU = [0.5, -0.2, 1.0]
const KF_H = [0.9 0.1 0.0
                0.0 0.8 0.2
                0.1 0.0 0.7]
const KF_F = [0.6 0.1 0.0
                0.0 0.5 0.1
                0.2 0.0 0.4]
const KF_Q_OBS = [0.30 0.05 0.00
                    0.05 0.25 0.02
                    0.00 0.02 0.20]
const KF_R_STATE = [0.40 0.00 0.05
                      0.00 0.35 0.00
                      0.05 0.00 0.30]
const KF_X0 = [0.0, 0.0, 0.0]
const KF_PX0 = Matrix{Float64}(I, 3, 3)

# Minimal model type implementing the Kalman API.
struct KFTestModel end
StateSpaceEcon.Kalman.kf_length_x(::KFTestModel) = 3
StateSpaceEcon.Kalman.kf_length_y(::KFTestModel) = 3
StateSpaceEcon.Kalman.kf_is_linear(::KFTestModel) = true
StateSpaceEcon.Kalman.kf_state_noise_shaping(::KFTestModel) = false
function StateSpaceEcon.Kalman.kf_linear_model(::KFTestModel)
    m = KFLinearModel(KFTestModel())
    m.mu .= KF_MU
    m.H .= KF_H
    m.F .= KF_F
    m.Q .= KF_Q_OBS
    m.R .= KF_R_STATE
    return m
end

# Wrapped in a runner (see runtests.jl) like the other e2e files.
function run_kalman_tests!()
@testset "Kalman filter / smoother" begin

    model = KFTestModel()

    # ----------------------------------------------------------------
    # API + container plumbing.
    # ----------------------------------------------------------------
    @testset "API + KFLinearModel construction" begin
        @test kf_length_x(model) == 3
        @test kf_length_y(model) == 3
        @test kf_is_linear(model) == true
        lm = kf_linear_model(model)
        @test lm isa KFLinearModel
        @test kf_length_x(lm) == 3
        @test kf_length_y(lm) == 3
        @test lm.mu == KF_MU
        @test lm.H == KF_H
        @test lm.F == KF_F
        # G is the identity UniformScaling when there is no noise shaping.
        @test lm.G === one(Float64) * I
    end

    @testset "KFData containers + @kfd_* macros" begin
        for KFDT in (KFDataFilter, KFDataFilterEx, KFDataSmoother,
                     KFDataSmootherEx)
            kfd = KFDT(1:12, model)
            @test eltype(kfd) == Float64
            @test size(kfd.x_pred) == (3, 12)
            @test size(kfd.Px_pred) == (3, 3, 12)
            @test size(kfd.x0) == (3,)
            # Round-trip a set / get / view through the macros.
            # `@kfd_set!` reads the value from a local of the same name.
            x_pred = [1.0, 2.0, 3.0]
            @kfd_set! kfd 5 x_pred
            @test (@kfd_get kfd 5 x_pred) == x_pred
            @test (@kfd_view kfd 5 x_pred) isa SubArray
            # Direct setindex! path.
            kfd[6, :x_pred] = [7.0, 8.0, 9.0]
            @test (@kfd_get kfd 6 x_pred) == [7.0, 8.0, 9.0]
        end
        # Integer RANGE works the same as a UnitRange.
        kfd_i = KFDataFilter(12, model)
        @test StateSpaceEcon.Kalman.kf_time_periods(kfd_i) == 12
    end

    @testset "KFilter workspace" begin
        kfd = KFDataFilter(1:12, model)
        kf = KFilter(kfd)
        @test kf isa KFilter
        @test eltype(kf) == Float64
        @test kf.kfd === kfd
        @test kf_length_x(kf) == 3
        @test kf_length_y(kf) == 3
        @test StateSpaceEcon.Kalman.kf_time_periods(kf) == 12
        # The data reference can be swapped for one with matching dims.
        kfd2 = KFDataFilterEx(1:24, model)
        kf.kfd = kfd2
        @test kf.kfd === kfd2
        @test StateSpaceEcon.Kalman.kf_time_periods(kf) == 24
    end

    # ----------------------------------------------------------------
    # Numerical match against the legacy reference.
    # ----------------------------------------------------------------
    ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                "kalman_reference.json"))
    if !isfile(ref_path)
        @info "skipping Kalman legacy match - capture_kalman.jl not run"
        @test_skip "kalman_reference.json not present"
    else
        ref = JSON.parsefile(ref_path)
        n = ref["n"]
        # Y is stored row-per-period.
        Y = Matrix{Float64}(undef, n, 3)
        for t in 1:n, j in 1:3
            Y[t, j] = Float64(ref["Y"][t][j])
        end
        @test ref["fwdstate"] == true

        # mat3: reconstruct a 3x3 from a column-major flattened JSON list.
        mat3(v) = reshape(Float64[Float64(x) for x in v], 3, 3)
        vec3(v) = Float64[Float64(x) for x in v]

        @testset "model constants match the reference" begin
            @test vec3(ref["model"]["mu"]) ≈ KF_MU
            @test mat3(ref["model"]["H"]) ≈ KF_H
            @test mat3(ref["model"]["F"]) ≈ KF_F
            @test mat3(ref["model"]["Q_obs"]) ≈ KF_Q_OBS
            @test mat3(ref["model"]["R_state"]) ≈ KF_R_STATE
        end

        # ---- Filter. ----
        kfd_f = kf_filter(Y, KF_X0, KF_PX0, model; fwdstate = true)

        @testset "filter trajectory matches legacy" begin
            # n=50 timesteps x 6 quantities -> 300 per-cell @tests; collapse
            # to one max-|Δ| assertion per quantity (6 asserts total).
            f = ref["filter"]
            d_x_pred  = 0.0; d_x       = 0.0; d_y_pred  = 0.0
            d_Px_pred = 0.0; d_Px      = 0.0; d_Py_pred = 0.0
            for t in 1:n
                d_x_pred  = max(d_x_pred,  maximum(abs.(kfd_f[t, :x_pred] .- vec3(f["x_pred"][t]))))
                d_x       = max(d_x,       maximum(abs.(kfd_f[t, :x]      .- vec3(f["x"][t]))))
                d_y_pred  = max(d_y_pred,  maximum(abs.(kfd_f[t, :y_pred] .- vec3(f["y_pred"][t]))))
                d_Px_pred = max(d_Px_pred, maximum(abs.(vec(kfd_f[t, :Px_pred]) .- vec3(f["Px_pred"][t]))))
                d_Px      = max(d_Px,      maximum(abs.(vec(kfd_f[t, :Px])      .- vec3(f["Px"][t]))))
                d_Py_pred = max(d_Py_pred, maximum(abs.(vec(kfd_f[t, :Py_pred]) .- vec3(f["Py_pred"][t]))))
            end
            @test d_x_pred  < 1e-10
            @test d_x       < 1e-10
            @test d_y_pred  < 1e-10
            @test d_Px_pred < 1e-10
            @test d_Px      < 1e-10
            @test d_Py_pred < 1e-10
        end

        @testset "filter log-likelihood matches legacy" begin
            f = ref["filter"]
            ll_ref = [Float64(x) for x in f["loglik"]]
            max_ll_diff = 0.0
            for t in 1:n
                max_ll_diff = max(max_ll_diff, abs(kfd_f[t, :loglik] - ll_ref[t]))
            end
            total_ll = sum(kfd_f[t, :loglik] for t in 1:n)
            @test max_ll_diff < 1e-10
            @test total_ll ≈ Float64(ref["total_loglik"])  atol=1e-9
            @info "Kalman log-likelihood match" total_ll max_per_period_diff=max_ll_diff
        end

        # ---- Smoother. ----
        kfd_s = kf_smoother(Y, KF_X0, KF_PX0, model; fwdstate = true)

        @testset "smoother trajectory matches legacy" begin
            # Same pattern as the filter testset: one max-|Δ| per quantity.
            s = ref["smoother"]
            d_x_s = 0.0; d_y_s = 0.0; d_Px_s = 0.0; d_Py_s = 0.0
            for t in 1:n
                d_x_s  = max(d_x_s,  maximum(abs.(kfd_s[t, :x_smooth] .- vec3(s["x_smooth"][t]))))
                d_y_s  = max(d_y_s,  maximum(abs.(kfd_s[t, :y_smooth] .- vec3(s["y_smooth"][t]))))
                d_Px_s = max(d_Px_s, maximum(abs.(vec(kfd_s[t, :Px_smooth]) .- vec3(s["Px_smooth"][t]))))
                d_Py_s = max(d_Py_s, maximum(abs.(vec(kfd_s[t, :Py_smooth]) .- vec3(s["Py_smooth"][t]))))
            end
            @test d_x_s  < 1e-10
            @test d_y_s  < 1e-10
            @test d_Px_s < 1e-10
            @test d_Py_s < 1e-10
        end

        # ---- Self-consistency invariants. ----
        @testset "filter / smoother covariance invariants" begin
            # Collapse 5 per-timestep checks across n=50 into worst-case
            # scalars; one assertion per invariant.
            sym_Px  = 0.0; sym_Pxs = 0.0
            min_ev_Px = Inf; min_ev_Pxs = Inf
            max_trace_excess = -Inf
            for t in 1:n
                Px = kfd_f[t, :Px]
                Pxs = kfd_s[t, :Px_smooth]
                sym_Px  = max(sym_Px,  maximum(abs.(Px  .- Px')))
                sym_Pxs = max(sym_Pxs, maximum(abs.(Pxs .- Pxs')))
                min_ev_Px  = min(min_ev_Px,  minimum(eigvals(Symmetric(Px))))
                min_ev_Pxs = min(min_ev_Pxs, minimum(eigvals(Symmetric(Pxs))))
                max_trace_excess = max(max_trace_excess, tr(Pxs) - tr(Px))
            end
            @test sym_Px  < 1e-12
            @test sym_Pxs < 1e-12
            @test min_ev_Px  > -1e-10
            @test min_ev_Pxs > -1e-10
            @test max_trace_excess <= 1e-9
        end
    end
end  # @testset
end  # function run_kalman_tests!
