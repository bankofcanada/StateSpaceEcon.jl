##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Cross-package numerical match against legacy ModelBaseEcon.jl /
# StateSpaceEcon.jl on simple_RBC.
#
# Reference data lives in the workspace at
# `legacy_reference/simple_RBC_reference.json`, captured by running
# `legacy_reference/capture.jl` against the legacy clones of
# ModelBaseEcon and StateSpaceEcon. This is the strict numerical-match
# deliverable for the simple_RBC end-to-end suite.
#
# Test approach: build the simple_RBC variant (plain `@variables`,
# no `@log` - see test/simple_rbc_e2e.jl for the rationale; equations
# have the same root set), solve SS and run the same impulse, then
# assert per-cell agreement with the legacy reference.
# ----------------------------------------------------------------------

using JSON

# Wrapped in a runner (see runtests.jl) so the suite can dispatch this
# self-contained testset on its own thread.
function run_legacy_reference_match!()
@testset "Cross-package match: simple_RBC vs legacy MBE/SSE" begin
    ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                "simple_RBC_reference.json"))
    if !isfile(ref_path)
        @info "skipping legacy reference match - capture.jl has not been run"
        @test_skip "legacy reference not present"
        return
    end

    ref = JSON.parsefile(ref_path)
    @test ref["model_name"] == "simple_RBC"
    @test ref["sim_var_order"] == ["C", "K", "L", "w", "r", "A"]
    @test ref["sim_shock_name"] == "ea"

    # Build the model (same equations as test/simple_rbc_e2e.jl).
    function build_simple_rbc()
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
        @equations m begin
            C[t+1] * (1 + g) = β * C[t] * (r[t+1] + 1 - δ)
            (L[t])^γ * C[t] = w[t]
            r[t] * (K[t-1] / (1 + g))^(1 - α) = α * A[t] * L[t]^(1 - α)
            w[t] * L[t]^(α) = (1 - α) * A[t] * (K[t-1] / (1 + g))^α
            @lin K[t] + C[t] = A[t] * (K[t-1] / (1 + g))^α * (L[t])^(1 - α) +
                               (1 - δ) * (K[t-1] / (1 + g))
            log(A[t]) = λ * log(A[t-1]) + ea[t]
        end
        return @initialize m
    end

    compiled = build_simple_rbc()
    var_order = [v.name for v in compiled.defs.vars]
    @test [string(s) for s in var_order] == ref["sim_var_order"]

    # ---- Steady state. ----
    ss_ref = ref["sstate"]                  # Dict{String, Float64}
    prob = SteadyStateProblem(compiled)
    x0 = [ss_ref[string(v)] * 0.95 for v in var_order]
    x_ss, converged, _ = sssolve!(prob; x0 = x0, tol = 1e-12, maxiter = 80)
    @test converged

    @testset "SS levels match legacy" begin
        max_ss_diff = 0.0
        for (i, v) in enumerate(var_order)
            max_ss_diff = max(max_ss_diff,
                              abs(x_ss[i] - Float64(ss_ref[string(v)])))
        end
        @info "simple_RBC SS max |Δ| vs legacy" diff=max_ss_diff
        @test max_ss_diff < 1e-8
    end

    # ---- Stacked-time simulation. ----
    sim_T = ref["sim_T"]                    # 40
    impulse = Float64(ref["sim_shock_value"])  # 0.01
    shock_period = ref["sim_shock_period"]  # 1 (1-based, into interior)

    plan = StackedTimePlan(compiled, sim_T)
    @test plan.maxlag == 1
    @test plan.maxlead == 1

    x_init = reshape(copy(x_ss), 1, 6)
    x_term = reshape(copy(x_ss), 1, 6)
    e_full = zeros(sim_T + plan.maxlag + plan.maxlead, 1)
    e_full[plan.maxlag + shock_period, 1] = impulse

    x_guess = repeat(reshape(x_ss, 1, 6), sim_T, 1)
    x_full, conv, iters = simulate!(plan;
        x_init, x_term, e_full, x_guess, tol = 1e-12, maxiter = 80)
    @test conv

    # Reference simulation matrix: list of T row-vectors, each length n_var.
    sim_ref_rows = ref["simulation"]        # Vector{Vector{Float64}}
    @test length(sim_ref_rows) == sim_T

    @testset "trajectory matches legacy at every (t, v)" begin
        # The capture script uses `fctype=fclevel` to match the
        # `x_term = SS` boundary, so trajectories should agree to
        # solver tolerance (1e-12) plus Newton-iteration round-off.
        # Empirically the achieved bound is ~3e-14; we test 1e-10.
        max_abs_diff = 0.0
        argmax_t = 0; argmax_v = 0
        for t in 1:sim_T
            row_ref = sim_ref_rows[t]       # Vector{Any} from JSON
            for v in 1:6
                d = abs(x_full[plan.maxlag + t, v] - Float64(row_ref[v]))
                if d > max_abs_diff
                    max_abs_diff = d
                    argmax_t = t; argmax_v = v
                end
            end
        end
        @info "max |Δ| across legacy match" diff=max_abs_diff t=argmax_t v=argmax_v
        @test max_abs_diff < 1e-10
    end
end  # @testset
end  # function run_legacy_reference_match!
