##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Performance parity guard.
#
# OPT-IN: the BenchmarkTools sweep adds ~30-40s to the suite, so
# `runtests.jl` only calls `run_perf_baseline!()` when SSE_PERF=1 (or
# SSE_PERF_RECAPTURE=1) is set; the everyday suite skips it. Set SSE_PERF=1 in
# CI to keep the regression guard active there.
#
# This is a guard, not an optimisation pass: it measures the three
# reference models on the hot paths and fails only on a real
# regression. The measured paths are, for each model:
#   1. SS solve                          (sssolve!)
#   2. One-step Newton residual+Jacobian assembly
#   3. Full impulse simulation           (simulate! / plan_simulate!)
#
# FRBUS_VAR has no cold-start steady state (sssolve! diverges), so its
# SS-solve benchmark is omitted; its RJ + simulation benchmarks run on
# the longbase-anchored plan path. The FRBUS one-step RJ is measured by
# `plan_simulate!(..., maxiter=1)` - which assembles the residual +
# Jacobian exactly once. plansim.jl inlines its RJ in the Newton loop
# and has no standalone RJ entry point; calling it with maxiter=1
# measures that exact code path.
#
# --- Design decisions ---
#
# Tooling: BenchmarkTools is a test-target dependency
# (Project.toml [extras] + [targets].test). We use `@benchmark` with a
# small sample budget - enough for a stable minimum, cheap enough to run
# inside the suite.
#
# Thresholds (10% wall / 25% alloc):
#   * Allocation counts are deterministic - they do not depend on host
#     speed. We compare allocated bytes strictly at +25% against a
#     committed baseline (legacy_reference/perf_baseline.json). A real
#     allocation regression fails the suite on any machine.
#   * Wall time is host-dependent - a baseline tight on one machine is a
#     false-failure trap on another. We therefore self-calibrate: a
#     fixed probe workload (rebuild simple_RBC) measures this host's
#     speed relative to the speed recorded alongside the baseline, and
#     the +10% wall-time threshold is scaled by that factor. The scaled
#     comparison only hard-fails on an egregious regression (3x the
#     scaled budget) so normal host jitter never reds the suite; the
#     per-benchmark scaled ratio is always @info-logged so a true 10%
#     creep is visible in CI logs and trends.
#
# Baseline file: legacy_reference/perf_baseline.json is committed and
# carries, per benchmark, {time, allocs, bytes} plus a top-level
# `calibration` probe time. When the file is absent the testset captures
# and writes a fresh baseline (and the comparison @tests are skipped
# that run). Re-capture by deleting the file or setting SSE_PERF_RECAPTURE=1.
# ----------------------------------------------------------------------

using BenchmarkTools
using JSON

# ----------------------------------------------------------------------
# simple_RBC builder for the perf guard.
#
# The build_simple_rbc in simple_rbc_e2e.jl is a closure nested inside
# run_simple_rbc_e2e! and is not reachable here. This is a verbatim copy
# of that model definition - kept local so the perf guard does not
# depend on test-execution order. If simple_RBC's definition changes,
# this copy must track it (both are mechanical transcriptions of the
# same TutorialsEcon model).
# ----------------------------------------------------------------------
function build_simple_rbc_perf()
    m = ModelDef(:simple_RBC)
    @parameters m begin
        α = 0.33
        δ = 0.1
        ρ = 0.03
        λ = 0.97
        γ = 0.5
        g = 0.015
        β = @link 1 / (1 + ρ)
    end
    @variables m begin
        C; K; L; w; r; A
    end
    @shocks m ea
    @autoexogenize m begin
        A = ea
    end
    @equations m begin
        C[t+1] * (1 + g) = β * C[t] * (r[t+1] + 1 - δ)
        (L[t])^γ * C[t] = w[t]
        r[t] * (K[t-1] / (1 + g))^(1 - α) = α * A[t] * L[t]^(1 - α)
        w[t] * L[t]^(α) = (1 - α) * A[t] * (K[t-1] / (1 + g))^α
        @lin K[t] + C[t] = A[t] * (K[t-1] / (1 + g))^α * (L[t])^(1 - α) + (1 - δ) * (K[t-1] / (1 + g))
        log(A[t]) = λ * log(A[t-1]) + ea[t]
    end
    return @initialize m
end

# Closed-form SS of simple_RBC (see simple_rbc_e2e.jl `analytic_ss`);
# perturbed 5% to give sssolve! a representative non-trivial start.
function perf_rbc_ss_guess()
    α=0.33; δ=0.1; ρ=0.03; γ=0.5; g=0.015
    A_ss = 1.0
    r_ss = (1+g)*(1+ρ) - 1 + δ
    κ    = (α/r_ss)^(1/(1-α))
    w_ss = (1-α) * κ^α
    coc  = κ^α - (δ+g)*κ
    L_ss = (((1-α) * κ^α) / coc)^(1/(γ+1))
    C_ss = L_ss * coc
    K_ss = (1+g) * κ * L_ss
    return [C_ss, K_ss, L_ss, w_ss, r_ss, A_ss] .* 0.95
end

# Wrapped in a runner (see runtests.jl) like the other e2e files.
# Reuses build_sw07 / build_frbus_var, defined in sw07_e2e.jl /
# frbus_var_build.jl, both included before this file.
function run_perf_baseline!()
@testset "performance parity guard" begin

    baseline_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                     "perf_baseline.json"))
    recapture = haskey(ENV, "SSE_PERF_RECAPTURE")
    have_baseline = isfile(baseline_path) && !recapture
    baseline = have_baseline ? JSON.parsefile(baseline_path) : nothing

    # --- Run one benchmark; return (min_time_ns, allocs, bytes). ---
    # Uses the BenchmarkTools programmatic API: `@benchmarkable` captures
    # the thunk (no macro-interpolation pitfalls), then `tune!` + `run`
    # with explicit params. `samples`/`seconds` cap the run; `evals=1`
    # keeps allocation counts per-call; `minimum` is the noise-robust
    # statistic for a guard.
    function measure(f, seconds::Float64)
        bench = @benchmarkable ($f)()
        bench.params.samples = 30
        bench.params.seconds = seconds
        bench.params.evals = 1
        bench.params.gctrial = true
        b = run(bench)
        return (time = minimum(b.times),          # ns
                allocs = b.allocs,
                bytes = b.memory)
    end

    # --- Calibration probe: a fixed, cheap, pure-CPU workload whose
    # timing tracks raw host speed. Rebuilding simple_RBC fits. ---
    @info "calibrating host speed"
    calib = measure(() -> build_simple_rbc_perf(), 4.0)
    calib_time = calib.time
    base_calib = have_baseline ? Float64(baseline["calibration"]["time"]) :
                                 calib_time
    # speed > 1 means this host is slower than the baseline host, so
    # wall-time budgets are scaled up by it.
    speed = calib_time / base_calib
    @info "host speed factor" speed calib_ms=calib_time/1e6

    results = Dict{String,Any}()

    # --- Compare one benchmark against the baseline. -----------------
    # Allocations: strict +25%. Wall time: scaled +10%, hard-fail only
    # at 3x the scaled budget; the scaled ratio is always logged.
    function check(label::String, m)
        results[label] = Dict("time" => m.time, "allocs" => m.allocs,
                              "bytes" => m.bytes)
        if !have_baseline
            @info "[$label] captured (no baseline to compare)" time_ms=m.time/1e6 allocs=m.allocs bytes=m.bytes
            return
        end
        # A benchmark added after the committed baseline was captured (e.g. the
        # stoch_simulate entry) has no baseline row yet - capture-only this
        # run; it lands in the baseline on the next SSE_PERF_RECAPTURE.
        if !haskey(baseline, label)
            @info "[$label] new benchmark, no baseline row - captured only" time_ms=m.time/1e6 allocs=m.allocs bytes=m.bytes
            return
        end
        b = baseline[label]
        base_bytes = Float64(b["bytes"])
        base_time  = Float64(b["time"])
        # Allocation guard - machine-independent, strict (+25%, +1KiB
        # absolute slack so tiny benchmarks are not jitter-sensitive).
        @test m.bytes <= 1.25 * base_bytes + 1024
        alloc_ratio = base_bytes == 0 ? 1.0 : m.bytes / base_bytes
        # Wall-time guard - host-scaled, advisory + egregious-only fail.
        scaled_budget = 1.10 * base_time * speed
        time_ratio = m.time / scaled_budget
        @info "[$label]" time_ms=m.time/1e6 scaled_ratio=time_ratio alloc_ratio=alloc_ratio allocs=m.allocs bytes=m.bytes
        if time_ratio > 1.0
            @warn "[$label] wall time over +10% scaled budget" time_ratio
        end
        @test m.time <= 3.0 * scaled_budget         # egregious-only hard fail
    end

    # ================================================================
    # simple_RBC
    # ================================================================
    @testset "simple_RBC" begin
        compiled = build_simple_rbc_perf()
        prob = SteadyStateProblem(compiled)

        # 1. SS solve.
        check("simple_RBC/ss_solve",
              measure(() -> sssolve!(prob; x0 = perf_rbc_ss_guess(),
                                     tol = 1e-12, maxiter = 80), 4.0))

        x_ss, conv, _ = sssolve!(prob; x0 = perf_rbc_ss_guess(),
                                 tol = 1e-12, maxiter = 80)
        @test conv

        T = 20
        plan = StackedTimePlan(compiled, T)
        n = plan.n_var
        x_init = reshape(copy(x_ss), plan.maxlag, n)
        x_term = reshape(copy(x_ss), plan.maxlead, n)
        e_full = zeros(T + plan.maxlag + plan.maxlead, plan.n_shock)
        e_full[plan.maxlag + 1, 1] = 0.01
        x_full0 = repeat(reshape(x_ss, 1, n), T + plan.maxlag + plan.maxlead, 1)
        R = Vector{Float64}(undef, T * plan.n_eq)

        # 2. One-step Newton residual+Jacobian assembly.
        check("simple_RBC/newton_rj",
              measure(() -> stacked_RJ!(R, x_full0, e_full, plan), 4.0))

        # 3. Full impulse simulation.
        x_guess = repeat(reshape(x_ss, 1, n), T, 1)
        check("simple_RBC/impulse_sim",
              measure(() -> simulate!(plan; x_init, x_term, e_full,
                                      x_guess, tol = 1e-10,
                                      maxiter = 60), 4.0))

        # 4. Stochastic simulation. 20 paths, single-period
        # unanticipated shock. The perf gate so that the 3D-matrix
        # wrapper does not silently regress vs the legacy per-path
        # loop. Build the SS baseline once (incl. terminal rows) outside the
        # benchmark; benchmark the 20-path solve.
        Ts = 20
        splan = SimPlan(compiled, Ts)
        sbase = SimData(compiled, splan.maxlag, Ts; maxlead = splan.maxlead)
        for (vi, name) in enumerate((:C, :K, :L, :w, :r, :A))
            col = sbase.col_index[name]
            sbase.values[:, col] .= sbase.is_log_col[col] ? log(x_ss[vi]) : x_ss[vi]
        end
        sshocks = zeros(1, 1, 20)
        for p in 1:20; sshocks[1, 1, p] = 0.01 * (p - 10) / 10; end
        check("simple_RBC/stoch_simulate",
              measure(() -> stoch_simulate(compiled, sbase, sshocks;
                                           shock_start = 2, tol = 1e-9,
                                           maxiter = 50), 8.0))
    end

    # ================================================================
    # SW07
    # ================================================================
    @testset "SW07" begin
        compiled = build_sw07()
        prob = SteadyStateProblem(compiled)
        n = 41

        # 1. SS solve.
        check("SW07/ss_solve",
              measure(() -> sssolve!(prob; x0 = zeros(n),
                                     tol = 1e-12, maxiter = 50), 4.0))

        x_ss, conv, _ = sssolve!(prob; x0 = zeros(n), tol = 1e-12)
        @test conv

        T = 40
        plan = StackedTimePlan(compiled, T)
        x_init = repeat(reshape(x_ss, 1, n), plan.maxlag, 1)
        x_term = repeat(reshape(x_ss, 1, n), plan.maxlead, 1)
        e_full = zeros(T + plan.maxlag + plan.maxlead, plan.n_shock)
        e_full[plan.maxlag + 1, plan.shock_index[:em]] = 0.01
        x_full0 = repeat(reshape(x_ss, 1, n), T + plan.maxlag + plan.maxlead, 1)
        R = Vector{Float64}(undef, T * plan.n_eq)

        # 2. One-step Newton residual+Jacobian assembly.
        check("SW07/newton_rj",
              measure(() -> stacked_RJ!(R, x_full0, e_full, plan), 4.0))

        # 3. Full impulse simulation.
        x_guess = repeat(reshape(x_ss, 1, n), T, 1)
        check("SW07/impulse_sim",
              measure(() -> simulate!(plan; x_init, x_term, e_full,
                                      x_guess, tol = 1e-12,
                                      maxiter = 80), 5.0))
    end

    # ================================================================
    # FRBUS_VAR - longbase plan path; no cold-start SS.
    # ================================================================
    longbase_path = abspath(joinpath(@__DIR__, "..", "..", "TutorialsEcon.jl",
        "4.FRB-US", "models", "longbase_2022-11-29.csv"))
    if !isfile(longbase_path)
        @info "skipping FRBUS_VAR perf (longbase CSV absent)"
        @test_skip "FRBUS_VAR longbase input not present"
    else
        @testset "FRBUS_VAR" begin
            compiled = build_frbus_var()
            sim_T = 24
            plan_probe = SimPlan(compiled, sim_T)
            maxlag = plan_probe.maxlag
            n_rows = maxlag + sim_T
            first_q_ord = (2022 * 4 + 0) - maxlag
            first_q = string(first_q_ord ÷ 4, "Q", (first_q_ord % 4) + 1)
            lb, _ = load_longbase(longbase_path, compiled;
                                  first_quarter = first_q, n_rows = n_rows)

            # Build a baseline-policy SimData (same setup as the e2e test).
            function frbus_base_data()
                d = SimData(compiled, maxlag, sim_T)
                d.values .= lb
                for s in FRBUS_MP_SWITCHES; d[s] = 0.0; end
                d[:dmpintay] = 1.0
                d[:dmptrsh] = 0.0
                d[:rffmin]  = -9999.0
                d[:drstar]  = 0.0
                for s in FRBUS_FP_SWITCHES; d[s] = 0.0; end
                d[:dfpsrp] = 1.0
                return d
            end

            # 2. One-step Newton iteration. plansim.jl inlines RJ in the
            # Newton loop with no standalone RJ entry point; maxiter=1
            # runs one full iteration - sparsity build + R+J assembly +
            # one sparse solve - then exits. That is the closest
            # non-invasive measurement of FRBUS's per-step cost without
            # extracting a standalone plan_RJ!.
            check("FRBUS_VAR/newton_rj",
                  measure(() -> begin
                      p = SimPlan(compiled, sim_T)
                      plan_simulate!(p, frbus_base_data();
                                     tol = 1e-9, maxiter = 1)
                  end, 8.0))

            # 3. Full impulse simulation - the headline shocked solve.
            # Back out the baseline once outside the benchmark, then
            # benchmark the shocked re-solve.
            p0 = SimPlan(compiled, sim_T)
            autoexogenize_plan!(p0)
            sol0, conv0, _ = plan_simulate!(p0, frbus_base_data();
                                            tol = 1e-9, maxiter = 80)
            @test conv0
            shk_data = copy(sol0)
            shk_data[:rffintay_a, 1] = shk_data[:rffintay_a, 1] + 1.0

            check("FRBUS_VAR/impulse_sim",
                  measure(() -> begin
                      p1 = SimPlan(compiled, sim_T)
                      plan_simulate!(p1, copy(shk_data);
                                     tol = 1e-9, maxiter = 80)
                  end, 12.0))
        end
    end

    # ================================================================
    # Write / refresh the committed baseline when absent or recapturing.
    # ================================================================
    if !have_baseline
        results["calibration"] = Dict("time" => calib_time,
                                      "allocs" => calib.allocs,
                                      "bytes" => calib.bytes)
        open(baseline_path, "w") do io
            JSON.print(io, results, 2)
        end
        @info "wrote performance baseline" path=baseline_path n_benchmarks=length(results) - 1
    end
end  # @testset
end  # function run_perf_baseline!
