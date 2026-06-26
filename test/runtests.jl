##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# TimeSeriesEcon is in [extras] + [targets].test, so the test driver
# resolves it from the General registry and the TimeSeriesEconExt
# extension activates automatically. If TimeSeriesEcon is unavailable in
# the resolved environment (e.g. CI without registry access), the
# TS-dependent tests fall through to `@test_skip` via the `_HAS_TSE`
# guard at the top of `timeseries_e2e.jl`.

using ModelBaseEcon
using StateSpaceEcon
using LinearAlgebra: norm
using SparseArrays: nnz
using Test

# ----------------------------------------------------------------------
# The suite runs two correctness witnesses:
#
#   1. "parity" - solver match tests on the symbolic-core internals:
#      steady-state, stacked-time, first-order, DFM EM, Kalman, shock
#      decomposition, stochastic simulation, plus end-to-end model builds.
#      Several of these assert per-cell agreement against a committed
#      numerical reference and self-skip when the reference data is absent.
#
#   2. "legacy compatibility" - behavioural claims on the public
#      compatibility surface (the `Plan` alias, the `simulate`/`solve!`
#      router, level final conditions, the linear-solver selector), so
#      existing call sites are exercised against the current internals.
#
# A third group, "@slope (red)", carries the steady-state slope/growth
# surface as `@test_broken` until that axis is implemented.
# ----------------------------------------------------------------------

@testset "StateSpaceEcon" begin

    @testset "parity" begin
        include("core_solver.jl")
        include("simple_rbc_e2e.jl")
        include("small_nk_e2e.jl")
        include("legacy_reference_match.jl")
        include("plansim.jl")
        include("sw07_e2e.jl")          # defines build_sw07 + e2e
        include("sw07_legacy_match.jl") # uses build_sw07
        include("timeseries_e2e.jl")    # TimeSeriesEcon extension
        include("pardiso_dispatch.jl")  # Pardiso extension dispatch
        include("diagnose_sstate.jl")   # SS convergence diagnostics
        include("steadystate_user_eqns_e2e.jl")
        include("firstorder_match.jl")  # first-order (QZ) solver
        include("dfm_match.jl")         # DFM EM solver legacy-match
        include("stoch_simulate.jl")    # stochastic simulation
        include("frbus_var_build.jl")   # defines build_frbus_var
        include("frbus_var_e2e.jl")     # FRBUS_VAR longbase e2e
        include("kalman.jl")            # Kalman filter / smoother
        include("shock_decomp.jl")      # shock decomposition
        include("perf_baseline.jl")     # performance parity guard (opt-in)

        run_core_solver!()
        run_simple_rbc_e2e!()
        run_small_nk_e2e!()
        run_legacy_reference_match!()
        run_plansim_tests!()
        run_sw07_e2e!()
        run_sw07_legacy_match!()
        run_timeseries_e2e!()
        run_pardiso_dispatch_tests!()
        run_diagnose_sstate_tests!()
        run_steadystate_user_eqns_tests!()
        run_firstorder_match_tests!()
        run_dfm_match_tests!()
        run_stoch_simulate_tests!()
        run_frbus_var_e2e!()
        run_kalman_tests!()
        run_shock_decomp_tests!()

        # Performance parity guard. Opt-in: the BenchmarkTools sweep adds
        # ~30-40s to the suite, too slow to pay on every run. Run it with
        # SSE_PERF=1 (or SSE_PERF_RECAPTURE=1, which implies it). Skipped
        # by default so the everyday suite stays fast; the guard still
        # runs in CI by setting the env var.
        if haskey(ENV, "SSE_PERF") || haskey(ENV, "SSE_PERF_RECAPTURE")
            run_perf_baseline!()
        else
            @info "perf guard skipped (set SSE_PERF=1 to run the benchmarks)"
        end
    end

    @testset "legacy compatibility" begin
        include("legacy_compat.jl")
        run_legacy_compat_tests!()
    end

    # The rate (growth) final conditions and the dynamic steady-state
    # references depend on the steady-state slope axis, which is not yet
    # implemented. They are carried as `@test_broken` so they are visibly
    # pending rather than silently skipped; this group flips green when
    # the slope axis lands.
    @testset "@slope (red)" begin
        @test_broken isdefined(StateSpaceEcon, :FCMatchSSRate)
        @test_broken isdefined(StateSpaceEcon, :FCConstRate)
        @test_broken isdefined(StateSpaceEcon, :fcslope)
        @test_broken isdefined(StateSpaceEcon, :fcrate)
        @test_broken isdefined(StateSpaceEcon, :fcnatural)
    end

end
