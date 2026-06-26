##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Plan-based stacked-time simulation: SimData, SimPlan, the exog/endo
# swap, and shock back-out.
#
# These exercise the machinery the FRBUS_VAR longbase protocol needs,
# on small analytic models where the answer is known in closed form.
# ----------------------------------------------------------------------

# Wrapped in a runner (see runtests.jl) so the suite can dispatch this
# self-contained testset on its own thread.
function run_plansim_tests!()
@testset "plan-based simulation" begin

    # ------------------------------------------------------------------
    # Default mask: SimPlan reproduces the base StackedTime.simulate!.
    # ------------------------------------------------------------------
    @testset "default mask matches StackedTime.simulate!" begin
        # y[t] = α y[t-1] + c + e[t], single shock impulse at t=1.
        m = ModelDef(:ar1p)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + c + e[t]
        end
        compiled = @initialize m
        T = 8
        y_ss = 1.0 / (1 - 0.5)

        # Reference via the base solver.
        base = StackedTimePlan(compiled, T)
        x_init = reshape([y_ss], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(T + 1, 1)
        e_full[2, 1] = 1.0
        x_ref, conv_ref, _ = simulate!(base; x_init, x_term, e_full)
        @test conv_ref

        # Same problem via SimPlan + plan_simulate!.
        plan = SimPlan(compiled, T)
        @test plan.maxlag == 1
        @test plan.maxlead == 0
        @test count(plan.is_unknown) == 1          # y is the only unknown
        data = SimData(compiled, plan.maxlag, T)
        data[:y, 0] = y_ss                          # initial condition (row 1)
        data[:e, 1] = 1.0                           # shock at t=1
        for t in 1:T; data[:y, t] = y_ss; end       # guess
        _, conv, iters = plan_simulate!(plan, data; tol = 1e-12)
        @test conv
        for t in 1:T
            @test data[:y, t] ≈ x_ref[1 + t, 1]  atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Exog/endo swap: back a shock out of a known variable path.
    # ------------------------------------------------------------------
    @testset "shock back-out via autoexogenize swap" begin
        # y[t] = α y[t-1] + e[t]. autoexogenize: y <-> e.
        # Given a target y-path, the backed-out shock is
        #   e[t] = y[t] - α y[t-1].
        m = ModelDef(:backout)
        @parameters m begin; α = 0.7; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @autoexogenize m begin
            y = e
        end
        @equations m begin
            y[t] = α * y[t-1] + e[t]
        end
        compiled = @initialize m
        T = 6

        plan = SimPlan(compiled, T)
        autoexogenize_plan!(plan)
        # After the swap: y exogenous, e endogenous.
        @test !plan.is_unknown[plan.col_index[:y]]
        @test plan.is_unknown[plan.col_index[:e]]

        data = SimData(compiled, plan.maxlag, T)
        # A target y-path (initial condition + interior), held fixed.
        y_path = [1.0, 1.3, 1.55, 1.7, 1.6, 1.45, 1.2]   # y[0..6]
        data[:y, 0] = y_path[1]
        for t in 1:T; data[:y, t] = y_path[t + 1]; end
        _, conv, _ = plan_simulate!(plan, data; tol = 1e-12)
        @test conv
        # Recovered shocks must satisfy e[t] = y[t] - α y[t-1].
        for t in 1:T
            expected = y_path[t + 1] - 0.7 * y_path[t]
            @test data[:e, t] ≈ expected  atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Round-trip: back out shocks, then re-simulate and recover baseline.
    # This is exactly the FRBUS sanity check (main.jl "Recover the
    # Baseline Case").
    # ------------------------------------------------------------------
    @testset "round-trip: back-out then re-simulate recovers baseline" begin
        m = ModelDef(:roundtrip)
        @parameters m begin; α = 0.6; β = 0.25; end
        @variables m begin; y; z; end
        @shocks m begin; ey; ez; end
        @autoexogenize m begin
            y = ey
            z = ez
        end
        compiled = @initialize m
        @equations m begin
            y[t] = α * y[t-1] + β * z[t] + ey[t]
            z[t] = α * z[t-1] + ez[t]
        end
        # (equations added after autoexog - order is fine, @initialize
        #  freezes the whole def.)
        compiled = @initialize m
        T = 5

        # --- Build an arbitrary baseline y/z path. ---
        baseline = SimData(compiled, 1, T)
        ypath = [0.5, 0.8, 1.1, 0.9, 0.7, 0.6]
        zpath = [0.2, 0.35, 0.4, 0.3, 0.25, 0.15]
        baseline[:y, 0] = ypath[1]; baseline[:z, 0] = zpath[1]
        for t in 1:T
            baseline[:y, t] = ypath[t + 1]
            baseline[:z, t] = zpath[t + 1]
        end

        # --- Step 1: back out the shocks. ---
        p0 = SimPlan(compiled, T)
        autoexogenize_plan!(p0)
        d0 = copy(baseline)
        _, conv0, _ = plan_simulate!(p0, d0; tol = 1e-12)
        @test conv0

        # --- Step 2: re-simulate with those shocks, variables endogenous. ---
        p1 = SimPlan(compiled, T)               # default mask (vars unknown)
        d1 = SimData(compiled, 1, T)
        # initial conditions from baseline
        d1[:y, 0] = ypath[1]; d1[:z, 0] = zpath[1]
        # shocks from the back-out
        for t in 1:T
            d1[:ey, t] = d0[:ey, t]
            d1[:ez, t] = d0[:ez, t]
        end
        _, conv1, _ = plan_simulate!(p1, d1; tol = 1e-12)
        @test conv1

        # The re-simulated path must match the original baseline.
        for t in 1:T
            @test d1[:y, t] ≈ ypath[t + 1]  atol=1e-9
            @test d1[:z, t] ≈ zpath[t + 1]  atol=1e-9
        end
    end

    # ------------------------------------------------------------------
    # Square-system guard: a non-square mask is rejected.
    # ------------------------------------------------------------------
    @testset "non-square unknown mask is rejected" begin
        m = ModelDef(:sq2)
        @parameters m begin; α = 0.5; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + e[t]
        end
        compiled = @initialize m
        plan = SimPlan(compiled, 4)
        endogenize!(plan, :e)                   # now 2 unknowns, 1 equation
        data = SimData(compiled, 1, 4)
        @test_throws ErrorException plan_simulate!(plan, data)
    end
end  # @testset
end  # function run_plansim_tests!
