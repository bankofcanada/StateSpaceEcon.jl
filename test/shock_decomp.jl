##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Shock decomposition tests.
#
# Verifies the stacked-time shock decomposition in
# StateSpaceEcon.jl/src/shock_decomp.jl against:
#   1. A legacy numerical reference (legacy_reference/shockdecomp_reference.json,
#      captured by legacy_reference/capture_shockdecomp.jl from the
#      legacy StateSpaceEcon stacked-time `shockdecomp`) - the
#      done-criterion: output matches legacy StateSpaceEcon to floating-point
#      tolerance on a small linear model.
#   2. Self-consistency invariants - every cell's source contributions
#      sum to (shocked - control); the linear model's `nonlinear` column
#      is ~0; the legacy source-column ordering is reproduced.
#
# The reference model is a fixed 3-variable / 3-shock all-lag linear
# model; this file rebuilds it identically.
# ----------------------------------------------------------------------

using JSON

# Build the small linear model - identical to the model in
# legacy_reference/capture_shockdecomp.jl (the source of the reference).
function build_sd_model()
    m = ModelDef(:sd_linear)
    @parameters m begin
        a_a = 0.6; a_b = 0.1
        b_a = 0.2; b_b = 0.5; b_c = 0.1
        c_b = 0.3; c_c = 0.4
    end
    @variables m begin
        a; b; c
    end
    @shocks m begin
        ea; eb; ec
    end
    @autoexogenize m begin
        a = ea
        b = eb
        c = ec
    end
    @equations m begin
        a[t] = a_a * a[t-1] + a_b * b[t-1] + ea[t]
        b[t] = b_a * a[t-1] + b_b * b[t-1] + b_c * c[t-1] + eb[t]
        c[t] = c_b * b[t-1] + c_c * c[t-1] + ec[t]
    end
    return @initialize m
end

# Wrapped in a runner (see runtests.jl) like the other e2e files.
function run_shock_decomp_tests!()
@testset "shock decomposition" begin

    compiled = build_sd_model()
    @test length(compiled.defs.vars) == 3
    @test length(compiled.defs.shocks) == 3

    sim_T = 12
    maxlag = 1

    # --- Control: the zero solution (no shocks, zero init). ---
    control = SimData(compiled, maxlag, sim_T)
    # values all zero already.

    # --- Shocked: build the shocked exogenous data, then solve. ---
    shocked_in = SimData(compiled, maxlag, sim_T)
    shocked_in[:ea, 1] = 0.10
    shocked_in[:eb, 2] = 0.05
    shocked_in[:ec, 3] = 0.08
    # Non-zero initial condition on `a` (row 1 = the maxlag init row).
    shocked_in.values[1, compiled_col(compiled, :a)] = 0.20

    plan = SimPlan(compiled, sim_T)
    shocked, conv, _ = plan_simulate!(plan, shocked_in; tol = 1e-12,
                                      maxiter = 50)
    @test conv

    # --- Decompose. ---
    p_decomp = SimPlan(compiled, sim_T)
    result = shock_decomp(p_decomp, control, shocked; tol = 1e-9)

    @testset "source column ordering matches legacy" begin
        # Legacy stacked-time shockdecomp: init, term, shocks..., nonlinear.
        @test result.source_names == [:init, :term, :ea, :eb, :ec, :nonlinear]
    end

    @testset "contributions sum to shocked - control" begin
        # 3 vars x sim_T timesteps; collapse to one max-|Δ| assertion.
        max_sum_diff = 0.0
        for v in (:a, :b, :c)
            M = result.contrib[v]
            for t in 1:sim_T
                got = sum(M[maxlag + t, :])
                want = shocked[v, t] - control[v, t]
                max_sum_diff = max(max_sum_diff, abs(got - want))
            end
        end
        @test max_sum_diff < 1e-10
    end

    @testset "linear model: nonlinear column is ~zero" begin
        nl_idx = length(result.source_names)         # last column
        for v in (:a, :b, :c)
            M = result.contrib[v]
            @test maximum(abs, M[:, nl_idx]) < 1e-9
        end
    end

    @testset "all-lag model: term column is zero" begin
        term_idx = 2
        for v in (:a, :b, :c)
            @test maximum(abs, result.contrib[v][:, term_idx]) < 1e-12
        end
    end

    # ----------------------------------------------------------------
    # Numerical match against the legacy reference.
    # ----------------------------------------------------------------
    ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                "shockdecomp_reference.json"))
    if !isfile(ref_path)
        @info "skipping shock-decomp legacy match - capture not run"
        @test_skip "shockdecomp_reference.json not present"
    else
        ref = JSON.parsefile(ref_path)
        @test ref["sim_T"] == sim_T
        @test ref["maxlag"] == maxlag
        @test ref["maxlead"] == 0
        ref_srcs = Symbol[Symbol(s) for s in ref["source_names"]]
        @test result.source_names == ref_srcs

        n_rows = maxlag + sim_T

        @testset "control / shocked solutions match legacy" begin
            # 3 vars x n_rows x 2 series = many cells; one max-|Δ| each.
            max_ctrl = 0.0; max_shk = 0.0
            for v in (:a, :b, :c)
                cref = Float64[Float64(x) for x in ref["control"][string(v)]]
                sref = Float64[Float64(x) for x in ref["shocked"][string(v)]]
                @test length(cref) == n_rows
                col = compiled_col(compiled, v)
                for r in 1:n_rows
                    max_ctrl = max(max_ctrl, abs(control.values[r, col] - cref[r]))
                    max_shk  = max(max_shk,  abs(shocked.values[r, col] - sref[r]))
                end
            end
            @test max_ctrl < 1e-10
            @test max_shk  < 1e-10
        end

        @testset "decomposition matrix matches legacy" begin
            # Full decomp matrix per variable: many thousands of cells.
            # Collapse to one max-|Δ| assertion across the whole sweep.
            max_diff = 0.0
            for v in (:a, :b, :c)
                sd_ref = ref["sd"][string(v)]        # n_rows x n_source
                M = result.contrib[v]
                @test length(sd_ref) == n_rows
                for r in 1:n_rows
                    row_ref = Float64[Float64(x) for x in sd_ref[r]]
                    @test length(row_ref) == length(result.source_names)
                    for k in eachindex(row_ref)
                        max_diff = max(max_diff, abs(M[r, k] - row_ref[k]))
                    end
                end
            end
            @info "shock-decomp legacy match" max_abs_diff=max_diff
            @test max_diff < 1e-9
        end
    end
end  # @testset
end  # function run_shock_decomp_tests!

# Helper - unified-column index of a variable/shock name in a model.
function compiled_col(model, name::Symbol)
    def = model.defs
    for (i, v) in pairs(def.vars)
        v.name === name && return i
    end
    for (i, s) in pairs(def.shocks)
        s.name === name && return length(def.vars) + i
    end
    error("compiled_col: no column named $name")
end
