##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Smets & Wouters (2007) cross-package numerical match against legacy
# ModelBaseEcon.jl / StateSpaceEcon.jl.
#
# Reference data lives in the workspace at
# `legacy_reference/SW07_reference.json`, captured by running
# `legacy_reference/capture.jl SW07` against the legacy clones on
# branches `codegen` (MBE) and `inf_deriv_1dsolvers` (SSE). The capture
# uses the `ss_impulse` protocol: solve SS, apply a unit `em` monetary
# impulse, simulate 40q with `fctype=fclevel` (terminal = SS).
#
# This file rebuilds SW07 via `build_sw07()` (defined in sw07_e2e.jl,
# already included earlier in runtests.jl), solves SS, runs the same
# impulse, and asserts per-cell agreement with the legacy reference.
#
# Index alignment: the reference `sim_var_order` is exactly the
# `@variables` declaration order in build_sw07 (a, b, c, cf, dc, ...),
# so the variable index == reference column index directly.
# ----------------------------------------------------------------------

using JSON

# Wrapped in a runner (see runtests.jl) so the suite can dispatch this
# self-contained testset on its own thread. Calls `build_sw07` defined
# in sw07_e2e.jl (included before this file).
function run_sw07_legacy_match!()
@testset "Cross-package match: SW07 vs legacy MBE/SSE" begin
    ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                "SW07_reference.json"))
    if !isfile(ref_path)
        @info "skipping SW07 legacy reference match - capture.jl has not been run"
        @test_skip "legacy reference not present"
        return
    end

    ref = JSON.parsefile(ref_path)
    @test ref["model_name"] == "SW07"
    @test ref["protocol"] == "ss_impulse"
    @test ref["fctype"] == "fclevel"
    @test ref["sim_shock_name"] == "em"
    @test length(ref["sim_var_order"]) == 41

    compiled = build_sw07()
    n = 41
    var_order = [v.name for v in compiled.defs.vars]
    @test [string(s) for s in var_order] == ref["sim_var_order"]

    # ---- Steady state. ----
    ss_ref = ref["sstate"]                  # Dict{String, Float64}
    prob = SteadyStateProblem(compiled)
    @test prob.n_var == n
    @test prob.n_eq == n
    x_ss, converged, _ = sssolve!(prob; x0 = zeros(n), tol = 1e-12, maxiter = 50)
    @test converged

    @testset "SS levels match legacy" begin
        # 41 vars x per-cell @test inflates the testset count without
        # adding signal beyond max |Δ|; collapse to one assertion.
        max_ss_diff = 0.0
        for (i, v) in enumerate(var_order)
            max_ss_diff = max(max_ss_diff,
                              abs(x_ss[i] - Float64(ss_ref[string(v)])))
        end
        @info "SW07 SS max |Δ| vs legacy" diff=max_ss_diff
        @test max_ss_diff < 1e-8
    end

    # ---- Stacked-time simulation. ----
    sim_T = ref["sim_T"]                       # 40
    impulse = Float64(ref["sim_shock_value"])  # 0.01
    shock_period = ref["sim_shock_period"]     # 1 (1-based, into interior)

    plan = StackedTimePlan(compiled, sim_T)
    # SW07 has pinf[t-3] (maxlag=3) and leads (maxlead=1).
    @test plan.maxlag == 3
    @test plan.maxlead == 1

    # Boundary blocks are the SS row repeated maxlag / maxlead times.
    x_init = repeat(reshape(x_ss, 1, n), plan.maxlag, 1)
    x_term = repeat(reshape(x_ss, 1, n), plan.maxlead, 1)
    e_full = zeros(sim_T + plan.maxlag + plan.maxlead, 7)
    em_idx = plan.shock_index[:em]
    e_full[plan.maxlag + shock_period, em_idx] = impulse

    x_guess = repeat(reshape(x_ss, 1, n), sim_T, 1)
    x_full, conv, iters = simulate!(plan;
        x_init, x_term, e_full, x_guess, tol = 1e-12, maxiter = 80)
    @test conv

    # Reference simulation matrix: list of T row-vectors, each length n_var.
    sim_ref_rows = ref["simulation"]           # Vector{Vector{Float64}}
    @test length(sim_ref_rows) == sim_T

    @testset "trajectory matches legacy at every (t, v)" begin
        # `fctype=fclevel` matches the solver's x_term = SS clamp, so
        # trajectories agree to solver tolerance plus Newton round-off.
        # 40q x 41 vars = 1640 cells - per-cell @test would inflate the
        # testset count without adding signal beyond max |Δ|.
        max_abs_diff = 0.0
        argmax_t = 0; argmax_v = 0
        for t in 1:sim_T
            row_ref = sim_ref_rows[t]          # Vector{Any} from JSON
            for v in 1:n
                d = abs(x_full[plan.maxlag + t, v] - Float64(row_ref[v]))
                if d > max_abs_diff
                    max_abs_diff = d
                    argmax_t = t; argmax_v = v
                end
            end
        end
        @info "max |Δ| across SW07 legacy match" diff=max_abs_diff t=argmax_t v=argmax_v
        @test max_abs_diff < 1e-10
    end
end  # @testset
end  # function run_sw07_legacy_match!
