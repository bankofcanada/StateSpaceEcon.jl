##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Legacy compatibility tests.
#
# These assert behavioural claims on the public compatibility surface
# documented in `src/compat.jl`: the `Plan` alias, the non-mutating
# `simulate` / in-place `solve!` router, the `use_pardiso` / `use_umfpack`
# linear-solver selector, and the level final-condition names
# (`FCNone` / `FCGiven` / `FCMatchSSLevel`, `setfc!`, `fclevel`).
#
# The previous public solver suite was written against an earlier plan /
# data API (a different `Plan` type over labelled time series, rate-based
# final conditions, and a multi-variant Newton driver). The current
# internals replace that surface, so this is a focused rewrite that
# exercises the same behaviours through the compatibility names rather
# than a verbatim port. Behaviours with no current binding (rate final
# conditions, the steady-state slope axis) are carried in `slope_red.jl`.
# ----------------------------------------------------------------------

# Wrapped in a runner (see runtests.jl) so the suite can dispatch this
# self-contained testset on its own thread.
function run_legacy_compat_tests!()
@testset "legacy compatibility surface" begin

    # ------------------------------------------------------------------
    # `Plan` is the public spelling of the plan type.
    # ------------------------------------------------------------------
    @testset "Plan is an alias of SimPlan" begin
        @test Plan === SimPlan
        m = ModelDef(:plan_alias)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        plan = Plan(compiled, 6)
        @test plan isa SimPlan
        @test plan.maxlag == 1
        @test plan.maxlead == 0
    end

    # ------------------------------------------------------------------
    # The `simulate` / `solve!` router over a plan and data.
    #
    # `simulate` copies the data and returns the solved copy (the input
    # is unchanged); `solve!` mutates in place and returns the same data.
    # Both must reach the same answer as the direct `plan_simulate!`.
    # ------------------------------------------------------------------
    @testset "simulate copies; solve! mutates" begin
        # y[t] = α y[t-1] + c + e[t], single unit shock at t=1.
        m = ModelDef(:router)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + c + e[t]
        end
        compiled = @initialize m
        T = 8
        y_ss = 1.0 / (1 - 0.5)

        # Closed form for the deviation from SS after a unit impulse:
        #   dy[t] = α^(t-1) for t >= 1.
        function make_data()
            d = SimData(compiled, 1, T)
            d[:y, 0] = y_ss
            d[:e, 1] = 1.0
            for t in 1:T; d[:y, t] = y_ss; end
            return d
        end

        plan = Plan(compiled, T)

        # Non-mutating: the input keeps its guess, the output is solved.
        input = make_data()
        guess_before = input[:y, T]
        out = simulate(plan, input; tol = 1e-12)
        @test out !== input
        @test input[:y, T] == guess_before          # input untouched
        for t in 1:T
            @test out[:y, t] ≈ y_ss + 0.5^(t - 1)  atol=1e-10
        end

        # In-place: the same data object is mutated and returned.
        data = make_data()
        ret = solve!(plan, data; tol = 1e-12)
        @test ret === data
        for t in 1:T
            @test data[:y, t] ≈ y_ss + 0.5^(t - 1)  atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # The exog/endo plan editing surface, reached through the public
    # plan type. Backing a shock out of a known path is the canonical
    # autoexogenize round-trip.
    # ------------------------------------------------------------------
    @testset "exogenize! / endogenize! on a public Plan" begin
        m = ModelDef(:exoswap)
        @parameters m begin; α = 0.7; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + e[t]
        end
        compiled = @initialize m
        T = 5

        plan = Plan(compiled, T)
        # Default mask: y unknown, e known.
        @test plan.is_unknown[plan.col_index[:y]]
        @test !plan.is_unknown[plan.col_index[:e]]
        # Swap: make e the unknown and y exogenous.
        exogenize!(plan, :y)
        endogenize!(plan, :e)
        @test !plan.is_unknown[plan.col_index[:y]]
        @test plan.is_unknown[plan.col_index[:e]]

        data = SimData(compiled, 1, T)
        ypath = [1.0, 1.3, 1.55, 1.7, 1.6, 1.45]   # y[0..5]
        data[:y, 0] = ypath[1]
        for t in 1:T; data[:y, t] = ypath[t + 1]; end
        solve!(plan, data; tol = 1e-12)
        # Recovered shocks satisfy e[t] = y[t] - α y[t-1].
        for t in 1:T
            @test data[:e, t] ≈ ypath[t + 1] - 0.7 * ypath[t]  atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Level final conditions on a forward-looking model.
    #
    # A pure-lead model needs a terminal condition. `setfc!` writes the
    # trailing rows; `fclevel` reads them back. With the steady state in
    # the terminal rows, every interior period sits at the steady state.
    # ------------------------------------------------------------------
    @testset "level final conditions: setfc! / fclevel" begin
        m = ModelDef(:fwdfc)
        @parameters m begin; β = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = β * y[t+1] + c
        end
        compiled = @initialize m
        T = 5
        y_ss = 1.0 / (1 - 0.5)

        plan = Plan(compiled, T)
        @test plan.maxlag == 0
        @test plan.maxlead == 1

        data = SimData(compiled, plan.maxlag, T; maxlead = plan.maxlead)
        for t in 1:T; data[:y, t] = y_ss; end       # guess
        # Write the steady state into the terminal (lead) rows.
        setfc!(data, FCGiven(), :y, y_ss)
        @test all(fclevel(data, :y) .≈ y_ss)

        solve!(plan, data; tol = 1e-12)
        for t in 1:T
            @test data[:y, t] ≈ y_ss  atol=1e-10
        end

        # FCMatchSSLevel writes the same terminal rows as FCGiven.
        data2 = SimData(compiled, plan.maxlag, T; maxlead = plan.maxlead)
        setfc!(data2, FCMatchSSLevel(), :y, y_ss)
        @test fclevel(data2, :y) == fclevel(data, :y)

        # FCNone leaves the terminal rows untouched.
        data3 = SimData(compiled, plan.maxlag, T; maxlead = plan.maxlead)
        before = copy(fclevel(data3, :y))
        setfc!(data3, FCNone(), :y, 99.0)
        @test fclevel(data3, :y) == before
    end

    # ------------------------------------------------------------------
    # Linear-solver selector. `use_pardiso` / `use_umfpack` name the
    # `linsolve` symbol; UMFPACK is always available, Pardiso routes
    # through the extension when the package is loaded.
    # ------------------------------------------------------------------
    @testset "use_pardiso / use_umfpack name the linsolve symbol" begin
        @test use_umfpack() === :umfpack
        @test use_pardiso() === :pardiso

        # The UMFPACK path solves a small model end to end.
        m = ModelDef(:linsel)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        T = 6
        y_ss = 1.0 / (1 - 0.5)
        plan = Plan(compiled, T)
        data = SimData(compiled, 1, T)
        data[:y, 0] = 0.0
        for t in 1:T; data[:y, t] = y_ss; end
        solve!(plan, data; tol = 1e-12, linsolve = use_umfpack())
        for t in 1:T
            @test data[:y, t] ≈ 0.5^t * (0.0 - y_ss) + y_ss  atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Model edits re-freeze through @initialize: adding an equation to a
    # def and re-initialising yields a model with the new equation.
    # ------------------------------------------------------------------
    @testset "model edit re-initialises" begin
        m = ModelDef(:edit)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; z; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + c
            z[t] = α * z[t-1] + e[t]
        end
        compiled = @initialize m
        @test length(compiled) == 2
        plan = Plan(compiled, 4)
        @test plan.n_eq == 2
        @test count(plan.is_unknown) == 2           # y and z are unknowns
    end

end  # @testset
end  # function run_legacy_compat_tests!
