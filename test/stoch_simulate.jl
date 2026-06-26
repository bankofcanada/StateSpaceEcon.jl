##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Stochastic simulation (`stoch_simulate`), legacy match +
# functional-coverage + performance gates.
#
# Reference data is captured from the legacy
# StateSpaceEcon.StackedTimeSolver.stoch_simulate by
# `legacy_reference/capture_stoch_simulate.jl` (run separately against the
# legacy clones) and committed to
# `legacy_reference/stoch_simulate_reference.json`. This test rebuilds the
# same simple_RBC on the ModelBaseEcon DSL, feeds the EXACT innovation matrix
# from the JSON into the 3D shock array, runs `stoch_simulate`, and asserts a
# per-path match at atol=1e-10.
#
# Paths are seed-stable, not algorithm-stable: we do
# NOT re-run any RNG - the `innov` matrix in the JSON is the exact draw, so
# the match is independent of the Julia RNG version.
#
# Result-range <-> SimData mapping (the crux):
#   The legacy result range is `first(shkrng)-maxlag : last(plan)`, dumped as
#   `full_nrow` rows. Within it the first `maxlag` rows are lag/initial and
#   the last `maxlead` rows are terminal (fcgiven = SS) - neither is solved.
#   So the interior horizon is `T = full_nrow - maxlag - maxlead`, the
#   SimData spans `maxlag + T + maxlead = full_nrow` rows, and dump row
#   `maxlag + t` is interior period `t`. The shock's interior period is
#   `shock_start_result - maxlag`.
#
# Per the per-cell-collapse convention the many-(t,var) comparisons
# collapse to one `max|Δ|` assertion each.
#
# CTarget invariant: this wraps `plan_simulate!` on assembled kernels only -
# ModelBaseEcon kernels are untouched.
# ----------------------------------------------------------------------

using JSON

# simple_RBC on the DSL - equations verbatim with simple_rbc_e2e.jl and the
# legacy capture script.
function build_stoch_rbc()
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

function _stoch_analytic_ss(; α=0.33, δ=0.1, ρ=0.03, λ=0.97, γ=0.5, g=0.015)
    A_ss = 1.0
    r_ss = (1+g)*(1+ρ) - 1 + δ
    κ    = (α/r_ss)^(1/(1-α))
    w_ss = (1-α) * κ^α
    coc  = κ^α - (δ+g)*κ
    L_ss = (((1-α) * κ^α) / coc)^(1/(γ+1))
    C_ss = L_ss * coc
    K_ss = (1+g) * κ * L_ss
    return (; C=C_ss, K=K_ss, L=L_ss, w=w_ss, r=r_ss, A=A_ss)
end

# Solve the SS and return the variable->level map (declaration order
# C,K,L,w,r,A) that the baseline is filled with.
function _stoch_rbc_ss(compiled)
    ss = _stoch_analytic_ss()
    prob = SteadyStateProblem(compiled)
    x0 = [ss.C, ss.K, ss.L, ss.w, ss.r, ss.A] .* 0.95
    x_ss, conv, _ = sssolve!(prob; x0 = x0, tol = 1e-12, maxiter = 80)
    @assert conv "simple_RBC SS did not solve"
    return Dict(:C=>x_ss[1], :K=>x_ss[2], :L=>x_ss[3],
                :w=>x_ss[4], :r=>x_ss[5], :A=>x_ss[6])
end

# Build a baseline SimData over `T` interior periods, every row (lag,
# interior, terminal) at the steady state in solver space.
function _stoch_ss_baseline(compiled, ssvals, T, maxlead)
    plan = SimPlan(compiled, T)
    baseline = SimData(compiled, plan.maxlag, T; maxlead = maxlead)
    for (name, val) in ssvals
        c = baseline.col_index[name]
        baseline.values[:, c] .= baseline.is_log_col[c] ? log(val) : val
    end
    return baseline, plan
end

_jnan(x) = x === nothing ? NaN : Float64(x)

function run_stoch_simulate_tests!()
@testset "stochastic simulation" begin

    compiled = build_stoch_rbc()
    ssvals = _stoch_rbc_ss(compiled)

    # ------------------------------------------------------------------
    # Legacy-reference match.
    # ------------------------------------------------------------------
    ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                "stoch_simulate_reference.json"))
    @testset "legacy stoch_simulate match (atol=1e-10)" begin
        if !isfile(ref_path)
            @test_skip "stoch_simulate_reference.json not present"
        else
            REF = JSON.parsefile(ref_path)
            for scen in ("single", "multi")
                s = REF[scen]
                mlag = Int(s["maxlag"]); mlead = Int(s["maxlead"])
                full = Int(s["full_nrow"])
                T = full - mlag - mlead

                baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, mlead)
                @test size(baseline.values, 1) == full   # mapping precondition

                ir = Int(s["innov_rows"]); ic = Int(s["innov_cols"])
                innov = reshape([_jnan(x) for x in s["innov"]], ir, ic)
                shocks = zeros(ir, 1, ic)
                for p in 1:ic, i in 1:ir
                    shocks[i, 1, p] = innov[i, p]
                end
                sstart = Int(s["shock_start_result"]) - mlag   # interior period

                res = stoch_simulate(compiled, baseline, shocks;
                                     shock_start = sstart, tol = 1e-9,
                                     maxiter = 50)
                @test count(isfailed, res.paths) == 0

                # Per-representative-path match on ea/C/A levels -- collapse to
                # one max|Δ| across all (rep_path, var, t) cells.
                dmax = 0.0
                for pk in s["rep_paths"]
                    p = Int(pk)
                    rp = s["rep"][string(p)]
                    rpath = res[p]
                    for sym in (:ea, :C, :A)
                        refv = [_jnan(x) for x in rp[string(sym)]]
                        for t in 1:T
                            dmax = max(dmax,
                                abs(level_value(rpath, sym, t) - refv[mlag + t]))
                        end
                    end
                end
                @test dmax < 1e-10

                # Path mean across all realisations vs legacy moments (A, C).
                mmax = 0.0
                for sym in ("A", "C")
                    mref = [_jnan(x) for x in s["moments"][sym]["mean"]]
                    sy = Symbol(sym)
                    for t in 1:T
                        mt = sum(level_value(res[p], sy, t) for p in 1:ic) / ic
                        mmax = max(mmax, abs(mt - mref[mlag + t]))
                    end
                end
                @test mmax < 1e-10
            end
        end
    end

    # ------------------------------------------------------------------
    # Seed/innovation reproducibility - same inputs => identical paths.
    # ------------------------------------------------------------------
    @testset "reproducibility: same innovations ⇒ identical paths" begin
        T = 20
        baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
        shocks = zeros(3, 1, 5)
        # deterministic, no RNG: a fixed pattern of innovations
        for p in 1:5, i in 1:3
            shocks[i, 1, p] = 0.001 * (p - 3) * i
        end
        r1 = stoch_simulate(compiled, baseline, shocks; shock_start = 2, tol = 1e-11)
        r2 = stoch_simulate(compiled, baseline, shocks; shock_start = 2, tol = 1e-11)
        dmax = 0.0
        for p in 1:5, t in 1:T, sym in (:C, :K, :A)
            dmax = max(dmax, abs(level_value(r1[p], sym, t) -
                                 level_value(r2[p], sym, t)))
        end
        @test dmax == 0.0
    end

    # ------------------------------------------------------------------
    # Zero-shock determinism - every path equals the baseline exactly.
    # ------------------------------------------------------------------
    @testset "zero shocks ⇒ paths equal baseline" begin
        T = 15
        baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
        shocks = zeros(2, 1, 4)            # all-zero innovations
        res = stoch_simulate(compiled, baseline, shocks; shock_start = 3, tol = 1e-12)
        @test count(isfailed, res.paths) == 0
        dmax = 0.0
        for p in 1:4, t in 1:T, name in (:C, :K, :L, :w, :r, :A)
            dmax = max(dmax, abs(level_value(res[p], name, t) -
                                 level_value(baseline, name, t)))
        end
        @test dmax == 0.0
    end

    # ------------------------------------------------------------------
    # Unanticipated timing - a single innovation at interior period k leaves
    # all periods < k at baseline (no pre-impact response) and moves A at k.
    # This is the defining property that separates unanticipated from
    # anticipated (which would move pre-impact periods through the lead).
    # ------------------------------------------------------------------
    @testset "unanticipated: no pre-impact response" begin
        T = 20
        k = 8                              # impact interior period
        baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
        shocks = zeros(1, 1, 1)
        shocks[1, 1, 1] = 0.01
        res = stoch_simulate(compiled, baseline, shocks; shock_start = k, tol = 1e-12)
        r = res[1]
        # pre-impact periods identical to baseline (all vars).
        pre = 0.0
        for t in 1:(k-1), name in (:C, :K, :L, :w, :r, :A)
            pre = max(pre, abs(level_value(r, name, t) -
                               level_value(baseline, name, t)))
        end
        @test pre == 0.0
        # A jumps at impact by the innovation: log A[k] = log A_ss + 0.01.
        @test log(level_value(r, :A, k)) - log(ssvals[:A]) ≈ 0.01  atol=1e-10
        # and there IS a response after impact (sanity: not a no-op).
        @test abs(level_value(r, :A, k+1) - ssvals[:A]) > 1e-6
    end

    # ------------------------------------------------------------------
    # Functional-coverage gate (user-required): the 3D matrix API must not
    # lose any legacy capability. Each sub-block maps 1:1 to a legacy feature.
    # ------------------------------------------------------------------
    @testset "functional coverage vs legacy" begin
        # (a) multi-period shock range over many periods.
        @testset "multi-period shock range" begin
            T = 25
            baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
            nper = 6
            shocks = zeros(nper, 1, 3)
            for p in 1:3, i in 1:nper
                shocks[i, 1, p] = 0.002 * (i - 3) * (p - 2)
            end
            res = stoch_simulate(compiled, baseline, shocks; shock_start = 2, tol = 1e-11)
            @test count(isfailed, res.paths) == 0
            # each path's A path differs from baseline somewhere in the range.
            moved = [maximum(abs(level_value(res[p], :A, t) - ssvals[:A])
                             for t in 1:T) for p in 1:3]
            @test moved[1] > 1e-8 && moved[3] > 1e-8       # nonzero-shock paths
        end

        # (b) >1 shock column on a 2-shock toy model.
        @testset "multiple shock columns" begin
            tm = ModelDef(:twoshock)
            @parameters tm begin; a = 0.5; b = 0.3; end
            @variables tm begin; x; y; end
            @shocks tm begin; ex; ey; end
            @equations tm begin
                x[t] = a * x[t-1] + ex[t]
                y[t] = b * y[t-1] + 0.2 * x[t] + ey[t]
            end
            tc = @initialize tm
            T = 10
            tplan = SimPlan(tc, T)
            base = SimData(tc, tplan.maxlag, T; maxlead = tplan.maxlead)  # SS=0
            shocks = zeros(1, 2, 2)
            shocks[1, 1, 1] = 0.1     # ex on path 1
            shocks[1, 2, 2] = 0.2     # ey on path 2
            res = stoch_simulate(tc, base, shocks; shock_start = 1, tol = 1e-12)
            @test count(isfailed, res.paths) == 0
            # path 1: ex shock moves x then y; path 2: ey shock moves y only.
            @test res[1][:x, 1] ≈ 0.1  atol=1e-10
            @test abs(res[2][:x, 1]) < 1e-12          # path 2 has no x shock
            @test res[2][:y, 1] ≈ 0.2  atol=1e-10
        end

        # (c) per-path independence: perturbing path i leaves path j unchanged.
        @testset "per-path independence" begin
            T = 12
            baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
            s1 = zeros(1, 1, 2); s1[1,1,1] = 0.01; s1[1,1,2] = 0.02
            s2 = copy(s1); s2[1,1,1] = 0.5        # perturb only path 1's shock
            r1 = stoch_simulate(compiled, baseline, s1; shock_start = 3, tol = 1e-11)
            r2 = stoch_simulate(compiled, baseline, s2; shock_start = 3, tol = 1e-11)
            # path 2 identical between the two runs; path 1 differs.
            d2 = maximum(abs(level_value(r1[2], :C, t) - level_value(r2[2], :C, t))
                         for t in 1:T)
            d1 = maximum(abs(level_value(r1[1], :C, t) - level_value(r2[1], :C, t))
                         for t in 1:T)
            @test d2 == 0.0
            @test d1 > 1e-6
        end

        # (d) failed-path marking: a path whose solve cannot converge is
        #     marked SimFailed and does not abort the batch.
        @testset "failed-path marking" begin
            T = 12
            baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
            shocks = zeros(1, 1, 3)
            shocks[1,1,1] = 0.01      # converges
            shocks[1,1,2] = 80.0      # absurd impulse: log A jumps by 80 =>
                                      # A approx e^80, Newton blows up (Inf/NaN) and
                                      # the path is caught and marked SimFailed.
            shocks[1,1,3] = 0.02      # converges
            # Generous maxiter so the small-shock paths converge cleanly; only
            # the absurd path 2 fails, and it does not abort the batch.
            res = stoch_simulate(compiled, baseline, shocks;
                                 shock_start = 2, tol = 1e-9, maxiter = 50)
            @test isfailed(res[2])
            @test res[2] isa SimFailed
            @test !isfailed(res[1]) && !isfailed(res[3])   # others still solved
        end

        # (e) empty shock range => n_path baseline copies (legacy early return).
        @testset "empty shock range passthrough" begin
            T = 8
            baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
            shocks = zeros(0, 1, 4)        # T_shk = 0
            res = stoch_simulate(compiled, baseline, shocks; tol = 1e-12)
            @test length(res) == 4
            @test count(isfailed, res.paths) == 0
            dmax = 0.0
            for p in 1:4, t in 1:T, name in (:C, :A)
                dmax = max(dmax, abs(level_value(res[p], name, t) -
                                     level_value(baseline, name, t)))
            end
            @test dmax == 0.0
        end
    end

    # ------------------------------------------------------------------
    # Input validation.
    # ------------------------------------------------------------------
    @testset "input validation" begin
        T = 8
        baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
        # wrong shock-column count
        @test_throws ErrorException stoch_simulate(compiled, baseline,
            zeros(1, 3, 2); shock_start = 1)
        # shock range past horizon
        @test_throws ErrorException stoch_simulate(compiled, baseline,
            zeros(2, 1, 2); shock_start = T)        # 2 periods from T overruns
        # shock_start < 1
        @test_throws ErrorException stoch_simulate(compiled, baseline,
            zeros(1, 1, 2); shock_start = 0)
    end

    # ------------------------------------------------------------------
    # Distribution sanity (linear-Gaussian): at the impact period the path
    # mean of log A approx log A_ss (innovations are zero-mean) and the spread is
    # of order the innovation scale. Loose - a sanity check, not a gate.
    # ------------------------------------------------------------------
    @testset "distribution sanity" begin
        T = 20
        baseline, _ = _stoch_ss_baseline(compiled, ssvals, T, 1)
        # deterministic symmetric innovations summing to zero (no RNG).
        npath = 8
        vals = [-0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02]
        shocks = zeros(1, 1, npath)
        for p in 1:npath; shocks[1, 1, p] = vals[p]; end
        k = 5
        res = stoch_simulate(compiled, baseline, shocks; shock_start = k, tol = 1e-11)
        @test count(isfailed, res.paths) == 0
        logA = [log(level_value(res[p], :A, k)) for p in 1:npath]
        meanlogA = sum(logA) / npath
        # symmetric zero-sum innovations => mean impact log A approx log A_ss.
        @test meanlogA - log(ssvals[:A]) ≈ 0.0  atol=1e-10
        # spread is of order the innovation scale (max |innov| = 0.02).
        @test maximum(abs.(logA .- log(ssvals[:A]))) ≈ 0.02  atol=1e-10
    end
end  # @testset
end  # function run_stoch_simulate_tests!
