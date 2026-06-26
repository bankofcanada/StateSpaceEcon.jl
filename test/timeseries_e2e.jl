##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# StateSpaceEcon <-> upstream TimeSeriesEcon bridge tests.
#
# The TimeSeriesEconExt extension (ext/TimeSeriesEconExt.jl) activates
# when `using TimeSeriesEcon` succeeds. Three blocks:
#   1. MVTSeries view round-trip over a small SimData.
#   2. simple AR(1) sim -> MVTSeries wrap (per-cell match against the raw
#      matrix at atol=0).
#   3. SW07 legacy-reference match through MVTSeries (atol=1e-10, same
#      target as `sw07_legacy_match.jl`).
#
# When TimeSeriesEcon is not loadable (no path-pin clone, install failure),
# each block emits `@test_skip` - the rest of the suite is unaffected.
# ----------------------------------------------------------------------

const _HAS_TSE = try
    @eval using TimeSeriesEcon
    true
catch err
    @info "timeseries_e2e: TimeSeriesEcon not loadable; tests will skip" err
    false
end

using JSON

function run_timeseries_e2e!()
@testset "TimeSeriesEcon bridge" begin

    if !_HAS_TSE
        @testset "TimeSeriesEcon unavailable" begin
            @test_skip "TimeSeriesEcon not available - extension tests skipped"
        end
        return
    end

    # ------------------------------------------------------------------
    # (1) MVTSeries <-> SimData view round-trip on a tiny synthetic model.
    # ------------------------------------------------------------------
    @testset "MVTSeries wraps SimData as a labelled view" begin
        m = ModelDef(:tse_view)
        @parameters m begin; α = 0.5; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + e[t]
        end
        compiled = @initialize m
        sd = SimData(compiled, 1, 4)        # maxlag=1, T=4 -> 5 rows x 2 cols
        sd.values[1, 1] = 7.0
        sd.values[2, 1] = 1.0
        sd.values[3, 2] = 0.25               # shock cell

        mvts = MVTSeries(sd, 2007Q1)
        @test mvts isa MVTSeries
        @test size(mvts) == (5, 2)
        # Column labels in the model's vars-then-shocks order.
        @test collect(keys(getfield(mvts, :columns))) == [:y, :e]
        # rangeof spans the maxlag init row + T sim rows starting at fd.
        @test rangeof(mvts) == 2007Q1:2008Q1  # 5 quarters (2007Q1..2008Q1)

        # View aliasing: read through MVTSeries reflects raw matrix.
        @test mvts[2007Q1, :y] == 7.0
        @test mvts[2007Q2, :y] == 1.0
        @test mvts[2007Q3, :e] == 0.25

        # Mutations through the MVTSeries flow back to SimData.values.
        mvts[2007Q4, :y] = 99.0
        @test sd.values[4, 1] == 99.0

        # And vice versa: write to SimData, read through MVTSeries.
        sd.values[5, 2] = -3.5
        @test mvts[2008Q1, :e] == -3.5
    end

    # ------------------------------------------------------------------
    # (2) AR(1) sim -> SimData -> MVTSeries view.
    # ------------------------------------------------------------------
    @testset "simulate! output retrieved as MVTSeries" begin
        m = ModelDef(:tse_ar1)
        @parameters m begin; α = 0.4; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        T = 6
        plan = StackedTimePlan(compiled, T)
        x_init = reshape([0.0], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(T + 1, 0)
        x_full, conv, _ = simulate!(plan; x_init, x_term, e_full)
        @test conv

        # `compiled` has no shocks - SimData has 1 column (y). Adopt
        # x_full as the SimData backing store.
        sd = SimData(compiled, plan.maxlag, T)
        sd.values .= x_full
        mvts = MVTSeries(sd, 2007Q1)
        @test size(mvts) == (T + plan.maxlag, 1)
        # MVTSeries is a labelled view over the same buffer - assert
        # bitwise equality across the trajectory in one shot.
        @test all(mvts[2007Q1 + t, :y] === x_full[t + 1, 1] for t in 0:T)
    end

    # ------------------------------------------------------------------
    # (3) SW07 legacy-reference match through MVTSeries.
    #     Same numerical gate as the direct match - atol=1e-10 cell-by-cell.
    # ------------------------------------------------------------------
    @testset "SW07 legacy match through MVTSeries (atol=1e-10)" begin
        ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                    "SW07_reference.json"))
        if !isfile(ref_path)
            @info "skipping SW07 MVTSeries match - reference JSON missing"
            @test_skip "legacy reference not present"
            return
        end

        ref = JSON.parsefile(ref_path)
        @test ref["model_name"] == "SW07"
        n = 41
        sim_T = ref["sim_T"]                       # 40
        impulse = Float64(ref["sim_shock_value"])  # 0.01
        shock_period = ref["sim_shock_period"]     # 1

        # Build the model + solve SS + run the em impulse - same logic
        # as sw07_legacy_match.jl, but the assertion path goes through
        # the MVTSeries wrapper.
        compiled = build_sw07()
        prob = SteadyStateProblem(compiled)
        x_ss, ss_conv, _ = sssolve!(prob; x0 = zeros(n),
                                    tol = 1e-12, maxiter = 50)
        @test ss_conv

        plan = StackedTimePlan(compiled, sim_T)
        x_init = repeat(reshape(x_ss, 1, n), plan.maxlag, 1)
        x_term = repeat(reshape(x_ss, 1, n), plan.maxlead, 1)
        e_full = zeros(sim_T + plan.maxlag + plan.maxlead, 7)
        em_idx = plan.shock_index[:em]
        e_full[plan.maxlag + shock_period, em_idx] = impulse
        x_guess = repeat(reshape(x_ss, 1, n), sim_T, 1)
        x_full, conv, _ = simulate!(plan;
            x_init, x_term, e_full, x_guess,
            tol = 1e-12, maxiter = 80)
        @test conv

        # Wrap the simulated buffer (T rows x n cols) plus boundary rows
        # in an MVTSeries; first sim row sits at fd0 = 2007Q1.
        sd = SimData(compiled, plan.maxlag, sim_T)
        # SimData has n_var + n_shock = 41 + 7 columns; place sim data
        # into the first n columns (vars), leave shocks zeroed.
        sd.values[:, 1:n] .= x_full[1:(plan.maxlag + sim_T), :]
        mvts = MVTSeries(sd, 2007Q1)
        @test size(mvts, 1) == plan.maxlag + sim_T
        @test size(mvts, 2) == n + 7

        # Compute the worst per-cell deviation by walking the trajectory
        # through the MVTSeries accessors (the point of this test), then
        # collapse to a single assertion. 40q x 41 vars = 1640 cells -
        # per-cell `@test` would inflate the testset count without adding
        # signal beyond `max |Δ|`.
        var_names = Symbol[v.name for v in compiled.defs.vars]
        sim_ref_rows = ref["simulation"]
        max_abs_diff = 0.0
        argmax_t = 0; argmax_v = 0
        for t in 1:sim_T
            row_ref = sim_ref_rows[t]
            for v in 1:n
                got = mvts[2007Q1 + (plan.maxlag + t - 1), var_names[v]]
                d = abs(got - Float64(row_ref[v]))
                if d > max_abs_diff
                    max_abs_diff = d
                    argmax_t = t; argmax_v = v
                end
            end
        end
        @info "SW07 MVTSeries-path max |Δ| vs legacy" diff=max_abs_diff t=argmax_t v=argmax_v
        @test max_abs_diff < 1e-10
    end
end  # @testset
end  # function
