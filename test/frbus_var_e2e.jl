##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# FRBUS_VAR end-to-end.
#
# FRBUS_VAR is the headline model: 284 endogenous variables, 83
# exogenous variables, 284 auto-shocks, 284 equations, maxlag=15,
# maxlead=0. It exercises the `Model{Vector{Equation}}` path,
# array-parameter scalarization, the DSL features (@d/@dlog,
# @exogenous, @autoshocks), and the plan-based simulation with the
# autoexogenize swap.
#
# Protocol (`longbase`, see legacy_reference/capture.jl + the FRBUS
# tutorial main.jl):
#   1. Load `longbase` historical data into a SimData spanning the
#      maxlag initial-condition rows + the 24q simulation range.
#   2. Set the monetary (dmpintay) and fiscal (dfpsrp) policy switches.
#   3. Back out the shocks: autoexogenize swap (variables exogenous,
#      shocks endogenous), solve -> `sol_0` (the baseline).
#   4. Shocked run: from `sol_0`, bump `rffintay_a` by 1 at the first
#      sim period, solve with the default mask -> `sol_1`.
#   5. Impulse = sol_1 - sol_0.
#
# `build_frbus_var()` is defined in frbus_var_build.jl (included first).
# ----------------------------------------------------------------------

using JSON

# Monetary / fiscal policy switch sets (from FRBUS set_policy.jl).
const FRBUS_MP_SWITCHES = (:dmpintay, :dmptay, :dmptlr, :dmpalt,
                           :dmpgen, :dmpex, :dmprr)
const FRBUS_FP_SWITCHES = (:dfpex, :dfpsrp, :dfpdbt)

"Set monetary policy: `switch` = 1, all other MP switches = 0, every row."
function frbus_set_mp!(data::SimData, switch::Symbol)
    for s in FRBUS_MP_SWITCHES
        data[s] = 0.0
    end
    data[switch] = 1.0
    return data
end

"Set fiscal policy: `switch` = 1, all other FP switches = 0, every row."
function frbus_set_fp!(data::SimData, switch::Symbol)
    for s in FRBUS_FP_SWITCHES
        data[s] = 0.0
    end
    data[switch] = 1.0
    return data
end

function run_frbus_var_e2e!()
@testset "FRBUS_VAR end-to-end" begin

    longbase_path = abspath(joinpath(@__DIR__, "..", "..", "TutorialsEcon.jl",
        "4.FRB-US", "models", "longbase_2022-11-29.csv"))
    ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
        "FRBUS_VAR_reference.json"))

    if !isfile(longbase_path) || !isfile(ref_path)
        @info "skipping FRBUS_VAR e2e - longbase CSV or reference JSON absent"
        @test_skip "FRBUS_VAR inputs not present"
        return
    end

    compiled = build_frbus_var()

    @testset "model builds with FRBUS_VAR shape" begin
        @test compiled isa CompiledModel
        @test length(compiled.defs.vars) == 284          # endogenous
        @test length(compiled.defs.shocks) == 367        # 83 exog + 284 _a
        @test length(exogenous_names(compiled.defs)) == 83
        @test length(compiled) == 284                    # equations
        @test compiled.eqns isa Vector{Equation}         # vector-of-equation path
    end

    # ---- Load reference + longbase. ----
    ref = JSON.parsefile(ref_path)
    @test ref["model_name"] == "FRBUS_VAR"
    @test ref["protocol"] == "longbase"
    @test ref["sim_T"] == 24
    sim_T = ref["sim_T"]
    var_order = ref["sim_var_order"]                     # 367 names
    @test length(var_order) == 367

    plan_probe = SimPlan(compiled, sim_T)
    maxlag = plan_probe.maxlag                           # 15
    @test maxlag == 15
    @test plan_probe.maxlead == 0
    n_rows = maxlag + sim_T

    # longbase rows: maxlag initial-condition quarters before 2022Q1.
    # 2022Q1 minus `maxlag` quarters.
    first_q_ord = (2022 * 4 + 0) - maxlag
    first_q = string(first_q_ord ÷ 4, "Q", (first_q_ord % 4) + 1)
    lb, _ = load_longbase(longbase_path, compiled;
                          first_quarter = first_q, n_rows = n_rows)
    @test size(lb) == (n_rows, plan_probe.n_col)

    # ---- Step 1: back out the shocks (baseline sol_0). ----
    base_data = SimData(compiled, maxlag, sim_T)
    base_data.values .= lb                               # longbase everywhere
    frbus_set_mp!(base_data, :dmpintay)
    base_data[:dmptrsh] = 0.0
    base_data[:rffmin]  = -9999.0
    base_data[:drstar]  = 0.0
    frbus_set_fp!(base_data, :dfpsrp)

    p0 = SimPlan(compiled, sim_T)
    autoexogenize_plan!(p0)                              # vars<->shocks swap
    sol0, conv0, iters0 = plan_simulate!(p0, base_data;
                                         tol = 1e-9, maxiter = 80)
    @test conv0
    @info "FRBUS baseline backed out" iters0

    # ---- Step 2: shocked run (sol_1). ----
    shk_data = copy(sol0)                                # start from baseline
    # rffintay_a += 1 at the first sim period (row maxlag+1).
    shk_data[:rffintay_a, 1] = shk_data[:rffintay_a, 1] + 1.0

    p1 = SimPlan(compiled, sim_T)                        # default mask
    sol1, conv1, iters1 = plan_simulate!(p1, shk_data;
                                         tol = 1e-9, maxiter = 80)
    @test conv1
    @info "FRBUS shocked simulation done" iters1

    # ---- Compare against the reference. ----
    # Reference keys: baseline / shocked / impulse, each a Tx367 matrix.
    # Reference column j == model variable/shock named var_order[j].
    base_ref = ref["baseline"]
    shk_ref  = ref["shocked"]
    @test length(base_ref) == sim_T

    # Reference column j <-> model name var_order[j]. Compare in level
    # space: the reference JSON stores levels, and `level_value` undoes
    # the @log solver-space transform.
    sym_of = Dict{Int,Symbol}()
    for (j, nm) in enumerate(var_order)
        sym = Symbol(nm)
        haskey(sol0.col_index, sym) && (sym_of[j] = sym)
    end
    @test length(sym_of) == 367

    # Relative-aware cell comparison: FRBUS levels span interest rates
    # O(1) up to nominal GDP O(1e4), so an absolute bound alone is
    # meaningless. Pass if |Δ| <= atol + rtol·|want|.
    function compare(sol, ref_rows, label)
        worst_rel = 0.0; wt = 0; wj = 0; worst_abs = 0.0
        for t in 1:sim_T
            row = ref_rows[t]
            for (j, sym) in sym_of
                got = level_value(sol, sym, t)
                want = Float64(row[j])
                d = abs(got - want)
                rel = d / (1 + abs(want))
                if rel > worst_rel
                    worst_rel = rel; wt = t; wj = j; worst_abs = d
                end
            end
        end
        worst_var = wj == 0 ? "(exact — no nonzero Δ)" : var_order[wj]
        @info "FRBUS $label worst cell" rel=worst_rel abs=worst_abs t=wt var=worst_var
        return worst_rel
    end

    @testset "baseline matches legacy reference" begin
        # The back-out fixes the variables to longbase and solves for the
        # shocks; reading those variables back reproduces longbase, and
        # the reference baseline is itself longbase-derived - so this is
        # exact by construction. Kept as a guard on the data plumbing.
        worst = compare(sol0, base_ref, "baseline")
        @test worst < 1e-9
    end

    @testset "shocked matches legacy reference" begin
        # The genuine numerical match: a full 284-equation stacked-time
        # solve. The target was atol=1e-8; the achieved relative
        # error is ~6e-12 - at floating-point tolerance despite FRBUS's
        # arithmetic depth. No relaxation needed.
        worst = compare(sol1, shk_ref, "shocked")
        @test worst < 1e-9
    end

    @testset "impulse: rff jumps at impact" begin
        # Economic sanity: the rffintay_a shock lifts rff ~+1 at impact.
        d_rff = level_value(sol1, :rff, 1) - level_value(sol0, :rff, 1)
        @test d_rff ≈ 1.0  atol=0.2
    end
end
end  # function run_frbus_var_e2e!
