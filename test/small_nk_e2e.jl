##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Small New Keynesian model end-to-end.
#
# TutorialsEcon has no small NK model (only simple_RBC, US_SW07, FRB-US -
# the latter two are too large for this scope), so this is a
# hand-written 10-equation textbook NK model. Variables are in
# log-deviations from steady state, so the unique SS is zero. All
# equations are linear -> SS solve in one Newton step, stacked-time also
# linear.
#
# Equations (canonical 3-equation NK + 3 stationary shock processes
# + 4 auxiliary definitions):
#   1. IS:               x[t]    = x[t+1] - (1/σ)(i[t] - π[t+1] - r_n[t])
#   2. NKPC:             π[t]    = β*π[t+1] + κ*x[t] + u_p[t]
#   3. Taylor rule:      i[t]    = ρ_i*i[t-1] + (1-ρ_i)*(φ_π*π[t] + φ_x*x[t]) + u_m[t]
#   4. Natural rate:     r_n[t]  = ρ_a*r_n[t-1] + e_a[t]
#   5. Cost-push shock:  u_p[t]  = ρ_p*u_p[t-1] + e_p[t]
#   6. Monetary shock:   u_m[t]  = ρ_m*u_m[t-1] + e_m[t]
#   7. Real rate (def):  rr[t]   = i[t] - π[t+1]
#   8. Ex-post rr (def): rr_ex[t]= i[t] - π[t]
#   9. Nominal y growth: gy[t]   = x[t] - x[t-1] + π[t]
#  10. Inflation lag:    pl[t]   = π[t-1]
# ----------------------------------------------------------------------

# Wrapped in a runner (see runtests.jl) so the suite can dispatch this
# self-contained testset on its own thread.
function run_small_nk_e2e!()
@testset "small NK end-to-end" begin

    function build_small_nk()
        m = ModelDef(:small_nk)
        @parameters m begin
            σ   = 1.0           # CRRA / inverse intertemporal elasticity
            β   = 0.99
            κ   = 0.1           # NKPC slope
            ρ_i = 0.7           # Taylor rule smoothing
            φ_π = 1.5           # inflation response
            φ_x = 0.125         # output gap response
            ρ_a = 0.9           # natural rate persistence
            ρ_p = 0.5           # cost-push persistence
            ρ_m = 0.5           # monetary shock persistence
        end
        @variables m begin
            "Output gap"
            x
            "Inflation"
            π
            "Nominal interest rate"
            i
            "Natural rate"
            r_n
            "Cost-push shock state"
            u_p
            "Monetary shock state"
            u_m
            "Real rate"
            rr
            "Ex-post real rate"
            rr_ex
            "Nominal output growth"
            gy
            "Lagged inflation"
            pl
        end
        @shocks m begin
            e_a; e_p; e_m
        end
        @equations m begin
            x[t]     = x[t+1] - (1/σ) * (i[t] - π[t+1] - r_n[t])
            π[t]     = β * π[t+1] + κ * x[t] + u_p[t]
            i[t]     = ρ_i * i[t-1] + (1 - ρ_i) * (φ_π * π[t] + φ_x * x[t]) + u_m[t]
            r_n[t]   = ρ_a * r_n[t-1] + e_a[t]
            u_p[t]   = ρ_p * u_p[t-1] + e_p[t]
            u_m[t]   = ρ_m * u_m[t-1] + e_m[t]
            rr[t]    = i[t] - π[t+1]
            rr_ex[t] = i[t] - π[t]
            gy[t]    = x[t] - x[t-1] + π[t]
            pl[t]    = π[t-1]
        end
        return @initialize m
    end

    @testset "build + initialize" begin
        compiled = build_small_nk()
        @test length(compiled) == 10
        @test length(compiled.defs.vars) == 10
        @test length(compiled.defs.shocks) == 3
        names = [v.name for v in compiled.defs.vars]
        @test names == [:x, :π, :i, :r_n, :u_p, :u_m, :rr, :rr_ex, :gy, :pl]
    end

    @testset "steady state is zero" begin
        compiled = build_small_nk()
        prob = SteadyStateProblem(compiled)
        @test prob.n_var == 10
        @test prob.n_eq == 10

        x_ss, converged, iters = sssolve!(prob; x0 = 0.1 * ones(10), tol = 1e-12)
        @test converged
        @test iters <= 3                # linear system -> 1 Newton step
        for v in 1:10
            @test abs(x_ss[v]) < 1e-10
        end
    end

    @testset "flat trajectory at SS=0 with no shocks" begin
        compiled = build_small_nk()
        T = 15
        plan = StackedTimePlan(compiled, T)
        @test plan.maxlag == 1
        @test plan.maxlead == 1

        x_ss = zeros(10)
        x_init = reshape(copy(x_ss), 1, 10)
        x_term = reshape(copy(x_ss), 1, 10)
        e_full = zeros(T + plan.maxlag + plan.maxlead, 3)
        x_full, converged, iters = simulate!(plan;
            x_init, x_term, e_full, tol = 1e-12, maxiter = 10)
        @test converged
        @test iters <= 3
        for t in 1:T, v in 1:10
            @test abs(x_full[plan.maxlag + t, v]) < 1e-10
        end
    end

    @testset "natural-rate impulse: r_n follows AR(1) exactly" begin
        # Equation 4 isolates r_n: r_n[t] = ρ_a*r_n[t-1] + e_a[t].
        # No other equation feeds back into r_n, so the AR(1) path is
        # exact regardless of how the rest of the model responds.
        compiled = build_small_nk()
        T = 25
        plan = StackedTimePlan(compiled, T)
        x_init = zeros(1, 10)
        x_term = zeros(1, 10)
        e_full = zeros(T + plan.maxlag + plan.maxlead, 3)
        impulse = 0.01
        e_a_idx = 1                              # shocks declared in order e_a, e_p, e_m
        e_full[plan.maxlag + 1, e_a_idx] = impulse

        x_full, converged, _ = simulate!(plan;
            x_init, x_term, e_full, tol = 1e-12, maxiter = 30)
        @test converged

        ρ_a = 0.9
        r_n_idx = 4                              # var ordering above
        for t in 1:T
            expected = ρ_a^(t-1) * impulse
            @test x_full[plan.maxlag + t, r_n_idx] ≈ expected  atol=1e-10
        end
        # And cost-push / monetary shock states stay at zero - those
        # equations are isolated from the natural-rate shock.
        for t in 1:T
            @test abs(x_full[plan.maxlag + t, 5]) < 1e-10  # u_p
            @test abs(x_full[plan.maxlag + t, 6]) < 1e-10  # u_m
        end
    end

    @testset "definitional consistency: rr, rr_ex, pl, gy match by construction" begin
        compiled = build_small_nk()
        T = 20
        plan = StackedTimePlan(compiled, T)
        x_init = zeros(1, 10)
        x_term = zeros(1, 10)
        e_full = zeros(T + plan.maxlag + plan.maxlead, 3)
        # Use a non-trivial shock mix so all variables move.
        e_full[plan.maxlag + 1, 1] = 0.005       # e_a
        e_full[plan.maxlag + 2, 2] = 0.003       # e_p, period 2
        e_full[plan.maxlag + 3, 3] = -0.002      # e_m, period 3

        x_full, converged, _ = simulate!(plan;
            x_init, x_term, e_full, tol = 1e-12, maxiter = 30)
        @test converged

        # Var indices.
        x_idx, π_idx, i_idx, rr_idx, rr_ex_idx, gy_idx, pl_idx = 1, 2, 3, 7, 8, 9, 10

        # rr[t]    = i[t] - π[t+1] for t = 1..T-1 (no t+1 data at T+1
        # within the interior; the simulator clamps to x_term=0).
        for t in 1:T-1
            row    = plan.maxlag + t
            row_p1 = plan.maxlag + t + 1
            expected = x_full[row, i_idx] - x_full[row_p1, π_idx]
            @test x_full[row, rr_idx] ≈ expected  atol=1e-10
        end
        # rr_ex[t] = i[t] - π[t]
        for t in 1:T
            row = plan.maxlag + t
            @test x_full[row, rr_ex_idx] ≈ x_full[row, i_idx] - x_full[row, π_idx]  atol=1e-10
        end
        # pl[t] = π[t-1]; at t=1, π[t-1] is the lag boundary (=0).
        @test x_full[plan.maxlag + 1, pl_idx] ≈ 0.0  atol=1e-10
        for t in 2:T
            row    = plan.maxlag + t
            row_m1 = plan.maxlag + t - 1
            @test x_full[row, pl_idx] ≈ x_full[row_m1, π_idx]  atol=1e-10
        end
        # gy[t] = x[t] - x[t-1] + π[t]; at t=1 use boundary x[0]=0.
        @test x_full[plan.maxlag + 1, gy_idx] ≈ x_full[plan.maxlag + 1, x_idx] +
              x_full[plan.maxlag + 1, π_idx]  atol=1e-10
        for t in 2:T
            row    = plan.maxlag + t
            row_m1 = plan.maxlag + t - 1
            expected = x_full[row, x_idx] - x_full[row_m1, x_idx] + x_full[row, π_idx]
            @test x_full[row, gy_idx] ≈ expected  atol=1e-10
        end
    end

    @testset "linearity superposition: combined shock = sum of individual shocks" begin
        # The whole model is linear, so simulate(e1 + e2) == simulate(e1) + simulate(e2)
        # to floating-point precision.
        compiled = build_small_nk()
        T = 12
        plan = StackedTimePlan(compiled, T)
        x_init = zeros(1, 10)
        x_term = zeros(1, 10)

        function run_with(e_full)
            x, conv, _ = simulate!(plan; x_init, x_term, e_full, tol = 1e-12, maxiter = 30)
            @test conv
            return x
        end

        e1 = zeros(T + plan.maxlag + plan.maxlead, 3)
        e1[plan.maxlag + 1, 1] = 0.01            # e_a impulse
        e2 = zeros(T + plan.maxlag + plan.maxlead, 3)
        e2[plan.maxlag + 2, 2] = -0.005          # e_p impulse at t=2
        e_sum = e1 .+ e2

        x1   = run_with(e1)
        x2   = run_with(e2)
        xsum = run_with(e_sum)

        # Compare interior unknowns only (boundary rows are inputs).
        rows = (plan.maxlag + 1):(plan.maxlag + T)
        @test maximum(abs, view(xsum, rows, :) .- (view(x1, rows, :) .+ view(x2, rows, :))) < 1e-10
    end

    @testset "Taylor rule smoothing: ρ_i pins lagged-i feedback" begin
        # With only a monetary shock (e_m), all forward-looking blocks
        # respond, but the i recursion has a clean structure:
        #   i[t] = ρ_i*i[t-1] + (1-ρ_i)*(φ_π*π[t] + φ_x*x[t]) + u_m[t]
        # We can verify equation 3 holds at each t by re-evaluating it.
        compiled = build_small_nk()
        T = 15
        plan = StackedTimePlan(compiled, T)
        x_init = zeros(1, 10)
        x_term = zeros(1, 10)
        e_full = zeros(T + plan.maxlag + plan.maxlead, 3)
        e_full[plan.maxlag + 1, 3] = 0.005       # e_m

        x_full, converged, _ = simulate!(plan;
            x_init, x_term, e_full, tol = 1e-12, maxiter = 30)
        @test converged

        ρ_i = 0.7; φ_π = 1.5; φ_x = 0.125
        x_idx, π_idx, i_idx, u_m_idx = 1, 2, 3, 6

        for t in 1:T
            row    = plan.maxlag + t
            row_m1 = plan.maxlag + t - 1            # uses lag boundary at t=1
            i_t    = x_full[row, i_idx]
            i_lag  = x_full[row_m1, i_idx]
            expected = ρ_i * i_lag +
                (1 - ρ_i) * (φ_π * x_full[row, π_idx] + φ_x * x_full[row, x_idx]) +
                x_full[row, u_m_idx]
            @test i_t ≈ expected  atol=1e-10
        end
    end
end  # @testset
end  # function run_small_nk_e2e!
