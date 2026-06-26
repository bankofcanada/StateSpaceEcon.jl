##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# simple_RBC end-to-end (internal-consistency strategy).
#
# The strict deliverable is numerical agreement with the legacy
# ModelBaseEcon/StateSpaceEcon. Reference capture is a separate
# follow-up task. Here we run the model and assert its internal
# invariants against analytic SS, flat-trajectory invariance, impulse
# decay, and linearization-at-SS equality with the nonlinear model.
#
# The model is a variant of TutorialsEcon's simple_RBC:
# `@logvariables` -> `@variables`, `@log` annotations dropped on
# equations (the symbolic->codegen pipeline doesn't apply log
# transforms, but `LHS = RHS` and `log(LHS) = log(RHS)` have the same
# root set, so SS and dynamics are identical - only Newton's path
# differs).
# ----------------------------------------------------------------------

# Wrapped in a runner (see runtests.jl) so the suite can dispatch this
# self-contained testset on its own thread.
function run_simple_rbc_e2e!()
@testset "simple_RBC end-to-end" begin

    # Build the model once and reuse across testsets.
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
            "Consumption"
            C
            "Capital Stock"
            K
            "Labour"
            L
            "Real Wage"
            w
            "Real Rental Rate"
            r
            "Technological shock"
            A
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

    # Closed-form SS, derived from the equations by hand:
    #   A_ss = 1
    #   r_ss = (1+g)*(1+ρ) - 1 + δ
    #   κ    = (α/r_ss)^(1/(1-α))               # = K_ss/((1+g)*L_ss)
    #   w_ss = (1-α) * κ^α
    #   C/L  = κ^α - (δ+g)*κ
    #   L_ss^(γ+1) = (1-α)*κ^α / (C/L)
    #   C_ss = L_ss * (C/L)
    #   K_ss = (1+g) * κ * L_ss
    function analytic_ss(; α=0.33, δ=0.1, ρ=0.03, λ=0.97, γ=0.5, g=0.015)
        A_ss = 1.0
        r_ss = (1+g)*(1+ρ) - 1 + δ
        κ    = (α/r_ss)^(1/(1-α))
        w_ss = (1-α) * κ^α
        coc  = κ^α - (δ+g)*κ                    # C/L
        L_ss = (((1-α) * κ^α) / coc)^(1/(γ+1))
        C_ss = L_ss * coc
        K_ss = (1+g) * κ * L_ss
        return (; C=C_ss, K=K_ss, L=L_ss, w=w_ss, r=r_ss, A=A_ss)
    end

    @testset "build + initialize" begin
        compiled = build_simple_rbc()
        @test length(compiled) == 6
        @test length(compiled.defs.vars) == 6
        @test length(compiled.defs.shocks) == 1
        # var ordering matters for indexing later
        names = [v.name for v in compiled.defs.vars]
        @test names == [:C, :K, :L, :w, :r, :A]
    end

    @testset "steady state matches analytic closed form" begin
        compiled = build_simple_rbc()
        ss = analytic_ss()
        prob = SteadyStateProblem(compiled)
        @test prob.n_var == 6
        @test prob.n_eq == 6

        # Start near SS to keep the (somewhat ill-conditioned) Newton on
        # rails - v1 sssolve! is plain Newton with no line search or LM.
        x0 = [ss.C, ss.K, ss.L, ss.w, ss.r, ss.A] .* 0.95
        x_ss, converged, iters = sssolve!(prob; x0 = x0, tol = 1e-12, maxiter = 80)
        @test converged
        @test x_ss[1] ≈ ss.C  atol=1e-9
        @test x_ss[2] ≈ ss.K  atol=1e-9
        @test x_ss[3] ≈ ss.L  atol=1e-9
        @test x_ss[4] ≈ ss.w  atol=1e-9
        @test x_ss[5] ≈ ss.r  atol=1e-9
        @test x_ss[6] ≈ ss.A  atol=1e-9

        R = zeros(6)
        ss_residual!(R, x_ss, prob)
        @test norm(R) < 1e-9
    end

    @testset "flat trajectory: x_init=x_term=ss, no shocks" begin
        compiled = build_simple_rbc()
        ss = analytic_ss()
        prob = SteadyStateProblem(compiled)
        x0 = [ss.C, ss.K, ss.L, ss.w, ss.r, ss.A] .* 0.95
        x_ss, converged, _ = sssolve!(prob; x0 = x0, tol = 1e-12, maxiter = 80)
        @test converged

        T = 12
        plan = StackedTimePlan(compiled, T)
        @test plan.maxlag == 1
        @test plan.maxlead == 1

        # x_init: 1x6 (one lag row), x_term: 1x6 (one lead row).
        x_init = reshape(copy(x_ss), 1, 6)
        x_term = reshape(copy(x_ss), 1, 6)
        e_full = zeros(T + plan.maxlag + plan.maxlead, 1)
        # Provide the SS trajectory as the initial guess so Newton sees a
        # zero-residual start - flat must round-trip exactly.
        x_guess = repeat(reshape(x_ss, 1, 6), T, 1)
        x_full, converged, iters = simulate!(plan;
            x_init, x_term, e_full, x_guess, tol = 1e-12, maxiter = 30)
        @test converged
        @test iters <= 3                        # already at solution

        # Every interior period equals SS.
        for t in 1:T, v in 1:6
            @test x_full[plan.maxlag + t, v] ≈ x_ss[v]  atol=1e-9
        end
    end

    @testset "impulse: A follows λ^t exactly; macro vars decay back" begin
        compiled = build_simple_rbc()
        ss = analytic_ss()
        prob = SteadyStateProblem(compiled)
        x0 = [ss.C, ss.K, ss.L, ss.w, ss.r, ss.A] .* 0.95
        x_ss, converged, _ = sssolve!(prob; x0 = x0, tol = 1e-12, maxiter = 80)
        @test converged

        T = 20
        plan = StackedTimePlan(compiled, T)
        x_init = reshape(copy(x_ss), 1, 6)
        x_term = reshape(copy(x_ss), 1, 6)
        e_full = zeros(T + plan.maxlag + plan.maxlead, 1)
        # Small impulse so we stay in a region where plain Newton converges.
        impulse = 0.01
        e_full[plan.maxlag + 1, 1] = impulse    # shock at t=1

        x_guess = repeat(reshape(x_ss, 1, 6), T, 1)
        x_full, converged, _ = simulate!(plan;
            x_init, x_term, e_full, x_guess, tol = 1e-10, maxiter = 60)
        @test converged

        # log(A[t]) = λ*log(A[t-1]) + ea[t]. With A[0]=1 (so log A[0]=0),
        # log A[1] = impulse, log A[t] = λ^(t-1) * impulse for t >= 1.
        λ = 0.97
        A_idx = 6
        for t in 1:T
            expected_logA = λ^(t-1) * impulse
            @test log(x_full[plan.maxlag + t, A_idx]) ≈ expected_logA  atol=1e-9
        end

        # Capital accumulation is slow (memory ~1/δ), so the macro
        # response does not have to be monotone in t over a short horizon.
        # What we can assert is that the response stays bounded - no
        # explosion - and is small in magnitude relative to the impulse.
        max_macro_dev = 0.0
        for t in 1:T, v in 1:5
            max_macro_dev = max(max_macro_dev, abs(x_full[plan.maxlag + t, v] - x_ss[v]))
        end
        # First-order response scales linearly with impulse; allow plenty
        # of slack since capital amplifies (κ ≈ 3.4x larger than L).
        @test max_macro_dev < 50 * impulse
    end

    @testset "selectively_linearize at SS: residual zero, flat trajectory preserved" begin
        compiled = build_simple_rbc()
        ss = analytic_ss()
        prob = SteadyStateProblem(compiled)
        x0 = [ss.C, ss.K, ss.L, ss.w, ss.r, ss.A] .* 0.95
        x_ss, _, _ = sssolve!(prob; x0 = x0, tol = 1e-12, maxiter = 80)

        # Equation 5 carries @lin. selectively_linearize must succeed
        # (R_ss for that equation must be ~0 at the SS solution).
        lin_model = selectively_linearize(compiled, x_ss; tol = 1e-7)
        @test length(lin_model) == 6

        # The non-@lin equations are kept by object identity; the @lin
        # one is replaced.
        for (i, (orig, lin)) in enumerate(zip(compiled.eqns, lin_model.eqns))
            if ModelBaseEcon.IR.EQ_LIN in orig.flags
                @test lin !== orig
            else
                @test lin === orig
            end
        end

        # Flat trajectory still flat under the linearized model.
        T = 8
        plan_lin = StackedTimePlan(lin_model, T)
        x_init = reshape(copy(x_ss), 1, 6)
        x_term = reshape(copy(x_ss), 1, 6)
        e_full = zeros(T + plan_lin.maxlag + plan_lin.maxlead, 1)
        x_guess = repeat(reshape(x_ss, 1, 6), T, 1)
        x_full, converged, _ = simulate!(plan_lin;
            x_init, x_term, e_full, x_guess, tol = 1e-12, maxiter = 30)
        @test converged
        for t in 1:T, v in 1:6
            @test x_full[plan_lin.maxlag + t, v] ≈ x_ss[v]  atol=1e-9
        end
    end

    @testset "linearized vs nonlinear: small-shock trajectories match closely" begin
        compiled = build_simple_rbc()
        ss = analytic_ss()
        prob = SteadyStateProblem(compiled)
        x0 = [ss.C, ss.K, ss.L, ss.w, ss.r, ss.A] .* 0.95
        x_ss, _, _ = sssolve!(prob; x0 = x0, tol = 1e-12, maxiter = 80)

        lin_model = selectively_linearize(compiled, x_ss; tol = 1e-7)

        T = 15
        plan_nl  = StackedTimePlan(compiled,  T)
        plan_lin = StackedTimePlan(lin_model, T)

        x_init = reshape(copy(x_ss), 1, 6)
        x_term = reshape(copy(x_ss), 1, 6)
        e_full = zeros(T + plan_nl.maxlag + plan_nl.maxlead, 1)
        impulse = 0.005                          # small enough for first-order to be tight
        e_full[plan_nl.maxlag + 1, 1] = impulse

        x_guess = repeat(reshape(x_ss, 1, 6), T, 1)
        x_nl,  conv_nl,  _ = simulate!(plan_nl;
            x_init, x_term, e_full, x_guess, tol = 1e-12, maxiter = 60)
        x_lin, conv_lin, _ = simulate!(plan_lin;
            x_init, x_term, e_full, x_guess, tol = 1e-12, maxiter = 60)
        @test conv_nl
        @test conv_lin

        # Only equation 5 is linearized; the rest are still nonlinear.
        # So we expect the trajectories to differ slightly but stay close.
        # Tolerance scales with impulse^2 ≈ 2.5e-5; pick 5e-4 to be safe
        # against accumulation over the horizon.
        max_diff = 0.0
        for t in 1:T, v in 1:6
            d = abs(x_nl[plan_nl.maxlag + t, v] - x_lin[plan_lin.maxlag + t, v])
            max_diff = max(max_diff, d)
        end
        @test max_diff < 5e-4
    end
end  # @testset
end  # function run_simple_rbc_e2e!
