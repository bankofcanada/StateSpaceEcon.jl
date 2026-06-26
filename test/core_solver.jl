##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Core solver tests: steady-state Newton and stacked-time simulation on
# small analytic models, plus a synthetic large-model build/simulate. Each
# block checks results against a closed-form solution.
# ----------------------------------------------------------------------

using LinearAlgebra: norm
using SparseArrays: nnz
using ModelBaseEcon: IR

function run_core_solver!()
    @testset "module loads" begin
        @test isdefined(StateSpaceEcon, :StateSpaceEcon)
    end

    # ------------------------------------------------------------------
    # Steady-state Newton on tiny analytic models.
    # ------------------------------------------------------------------

    @testset "sssolve: scalar AR(1) - analytic SS" begin
        # y[t] = α*y[t-1] + c
        # SS: y_ss = α*y_ss + c -> y_ss = c/(1-α).
        m = ModelDef(:ar1)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        prob = SteadyStateProblem(compiled)
        @test prob.n_var == 1
        @test prob.n_eq == 1

        x_ss, converged, iters = sssolve!(prob; x0 = [0.5], tol = 1e-12)
        @test converged
        @test x_ss[1] ≈ 1.0 / (1 - 0.5)  atol=1e-10
        # Residual at solution is ~0.
        R = zeros(1)
        ss_residual!(R, x_ss, prob)
        @test abs(R[1]) < 1e-10
    end

    @testset "sssolve: 2-equation linear system" begin
        # y[t] = a*y[t-1] + b*z[t]
        # z[t] = c*z[t-1] + d
        # SS: y = a*y + b*z, z = c*z + d
        #     z_ss = d/(1-c), y_ss = b*z_ss / (1-a)
        m = ModelDef(:lin2)
        @parameters m begin
            a = 0.3; b = 0.4; c = 0.6; d = 1.0
        end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = a * y[t-1] + b * z[t]
            z[t] = c * z[t-1] + d
        end
        compiled = @initialize m
        prob = SteadyStateProblem(compiled)
        x_ss, converged, _ = sssolve!(prob; x0 = [0.0, 0.0])
        @test converged
        z_ss_expected = 1.0 / (1 - 0.6)
        y_ss_expected = 0.4 * z_ss_expected / (1 - 0.3)
        @test x_ss[1] ≈ y_ss_expected  atol=1e-10
        @test x_ss[2] ≈ z_ss_expected  atol=1e-10
    end

    @testset "sssolve: nonlinear scalar (Newton convergence)" begin
        # y[t]^2 = c -> y_ss = √c (positive root if x0 > 0).
        m = ModelDef(:sq)
        @parameters m begin; c = 4.0; end
        @variables m begin; y; end
        @equations m begin
            y[t]^2 = c
        end
        compiled = @initialize m
        prob = SteadyStateProblem(compiled)
        x_ss, converged, iters = sssolve!(prob; x0 = [3.0])
        @test converged
        @test x_ss[1] ≈ 2.0  atol=1e-10
        @test iters < 20
    end

    @testset "sssolve: shocks treated as zero at SS" begin
        # y[t] = α*y[t-1] + c + e[t]
        # At SS, e=0: y_ss = c/(1-α).
        m = ModelDef(:shocked)
        @parameters m begin; α = 0.4; c = 2.0; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + c + e[t]
        end
        compiled = @initialize m
        prob = SteadyStateProblem(compiled)
        x_ss, converged, _ = sssolve!(prob; x0 = [0.0])
        @test converged
        @test x_ss[1] ≈ 2.0 / (1 - 0.4)  atol=1e-10
    end

    @testset "sssolve: lead/lag time refs collapse correctly at SS" begin
        # y[t+1] - β*y[t] - β*y[t-1] - c = 0
        # SS: y - β*y - β*y - c = 0 -> y(1 - 2β) = c -> y = c/(1-2β)
        m = ModelDef(:lead_lag)
        @parameters m begin; β = 0.3; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t+1] = β * y[t] + β * y[t-1] + c
        end
        compiled = @initialize m
        prob = SteadyStateProblem(compiled)
        x_ss, converged, _ = sssolve!(prob; x0 = [0.0])
        @test converged
        @test x_ss[1] ≈ 1.0 / (1 - 0.6)  atol=1e-10
    end

    @testset "sssolve: link parameter resolves through SS" begin
        # β = @link 1/(1+ρ); equation: y[t] = β*y[t-1] + c
        # SS: y = β*y + c -> y = c/(1-β) = c*(1+ρ)/ρ
        m = ModelDef(:linked)
        @parameters m begin
            ρ = 0.05
            c = 1.0
            β = @link 1 / (1 + ρ)
        end
        @variables m begin; y; end
        @equations m begin
            y[t] = β * y[t-1] + c
        end
        compiled = @initialize m
        prob = SteadyStateProblem(compiled)
        x_ss, converged, _ = sssolve!(prob; x0 = [10.0])
        @test converged
        ρ = 0.05
        @test x_ss[1] ≈ 1.0 * (1 + ρ) / ρ  atol=1e-8
    end

    @testset "sssolve: rejects non-square system" begin
        # 2 vars, 1 equation -> underdetermined.
        m = ModelDef()
        @parameters m begin; α = 0.5; end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = α * y[t-1] + z[t]
        end
        compiled = @initialize m
        @test_throws ErrorException SteadyStateProblem(compiled)
    end

    # ------------------------------------------------------------------
    # Stacked-time Newton.
    # ------------------------------------------------------------------

    @testset "simulate: AR(1) propagates from initial condition" begin
        # y[t] = α*y[t-1] + c
        # Closed form: y[t] = α^t * (y_0 - y_ss) + y_ss, with y_ss = c/(1-α).
        m = ModelDef(:ar1)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        T = 10
        plan = StackedTimePlan(compiled, T)
        @test plan.maxlag == 1
        @test plan.maxlead == 0

        y_ss = 1.0 / (1 - 0.5)
        y_0 = 0.0
        # No leads -> no terminal condition needed.
        x_init = reshape([y_0], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(T + 1, 0)        # no shocks declared
        x_full, converged, iters = simulate!(plan; x_init, x_term, e_full)
        @test converged
        @test iters <= 3                # linear -> 1 Newton step (+ check)
        # Compare against closed form.
        for t in 1:T
            expected = 0.5^t * (y_0 - y_ss) + y_ss
            @test x_full[1 + t, 1] ≈ expected  atol=1e-10
        end
    end

    @testset "simulate: AR(1) with shock impulse" begin
        # y[t] = α*y[t-1] + c + e[t]
        # With y_0 = y_ss and a single unit shock at period 1, the deviation
        # from SS follows: dy[t] = α^(t-1) * 1 for t>=1.
        m = ModelDef(:ar1s)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + c + e[t]
        end
        compiled = @initialize m
        T = 8
        plan = StackedTimePlan(compiled, T)
        y_ss = 1.0 / (1 - 0.5)
        x_init = reshape([y_ss], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(T + 1, 1)        # rows: 1 (lag) + T
        e_full[2, 1] = 1.0              # shock at t=1 (row 2 = t=1 since maxlag=1)
        x_full, converged, _ = simulate!(plan; x_init, x_term, e_full)
        @test converged
        for t in 1:T
            expected = y_ss + 0.5^(t - 1) * 1.0
            @test x_full[1 + t, 1] ≈ expected  atol=1e-10
        end
    end

    @testset "simulate: forward-looking equation needs terminal condition" begin
        # Pure forward: y[t] = β * y[t+1] + c.
        # SS: y = β*y + c -> y_ss = c/(1-β).
        # With y[T+1] = y_ss as terminal condition, every interior y = y_ss.
        m = ModelDef(:fwd)
        @parameters m begin; β = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = β * y[t+1] + c
        end
        compiled = @initialize m
        T = 5
        plan = StackedTimePlan(compiled, T)
        @test plan.maxlag == 0
        @test plan.maxlead == 1

        y_ss = 1.0 / (1 - 0.5)
        x_init = zeros(0, 1)
        x_term = reshape([y_ss], 1, 1)
        e_full = zeros(T + 1, 0)
        x_full, converged, _ = simulate!(plan; x_init, x_term, e_full)
        @test converged
        for t in 1:T
            @test x_full[t, 1] ≈ y_ss  atol=1e-10
        end
    end

    @testset "simulate: 2-equation lead+lag system at SS stays at SS" begin
        # Two equations, one with lag, one with lead. Initial = terminal = SS.
        # Whole trajectory should be flat at SS.
        m = ModelDef(:flat)
        @parameters m begin
            α = 0.4; β = 0.5; c1 = 1.0; c2 = 2.0
        end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = α * y[t-1] + c1
            z[t] = β * z[t+1] + c2
        end
        compiled = @initialize m
        T = 6
        plan = StackedTimePlan(compiled, T)
        @test plan.maxlag == 1
        @test plan.maxlead == 1

        y_ss = c1 = 1.0; α = 0.4
        y_ss = c1 / (1 - α)
        β = 0.5; c2 = 2.0
        z_ss = c2 / (1 - β)

        x_init = reshape([y_ss z_ss], 1, 2)
        x_term = reshape([y_ss z_ss], 1, 2)
        e_full = zeros(T + 2, 0)
        x_full, converged, _ = simulate!(plan;
            x_init, x_term, e_full,
            x_guess = repeat([y_ss z_ss], T, 1))
        @test converged
        for t in 1:T
            @test x_full[1 + t, 1] ≈ y_ss  atol=1e-10
            @test x_full[1 + t, 2] ≈ z_ss  atol=1e-10
        end
    end

    @testset "simulate: nonlinear convergence" begin
        # y[t]^2 = α * y[t-1] + c, with y > 0.
        # SS: y_ss^2 = α*y_ss + c -> y_ss = (α + √(α²+4c))/2.
        m = ModelDef(:nlin)
        @parameters m begin; α = 0.3; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t]^2 = α * y[t-1] + c
        end
        compiled = @initialize m
        T = 5
        plan = StackedTimePlan(compiled, T)
        y_ss = (0.3 + sqrt(0.09 + 4.0)) / 2
        x_init = reshape([y_ss], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(T + 1, 0)
        x_full, converged, iters = simulate!(plan;
            x_init, x_term, e_full,
            x_guess = fill(y_ss, T, 1))
        @test converged
        @test iters <= 10
        for t in 1:T
            @test x_full[1 + t, 1] ≈ y_ss  atol=1e-9
        end
    end

    @testset "simulate: sparsity pattern is correctly built" begin
        # AR(1): one var, one eq, two slots (y[t-1], y[t]).
        # Pattern: tridiagonal-ish - each row t has nz at col t (y[t]) and
        # col t-1 (y[t-1]) when t-1 >= 1.
        m = ModelDef()
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        T = 4
        plan = StackedTimePlan(compiled, T)
        # n_rows = 4, n_cols = 4. Row 1 has only (1,1) [y[t-1]=initial cond].
        # Rows 2..4 have (t, t-1) and (t, t).
        @test size(plan.J) == (T, T)
        # Total nz = 1 (row1) + 2 + 2 + 2 = 7.
        @test nnz(plan.J) == 7
    end

    @testset "ss_RJ!: jacobian structure for 2-eq system" begin
        # Same lin2 model. Verify J is the SS-collapsed gradient.
        m = ModelDef()
        @parameters m begin
            a = 0.3; b = 0.4; c = 0.6; d = 1.0
        end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = a * y[t-1] + b * z[t]    # F1 = y - a*y_lag - b*z
            z[t] = c * z[t-1] + d           # F2 = z - c*z_lag - d
        end
        compiled = @initialize m
        prob = SteadyStateProblem(compiled)
        x = [1.0, 1.0]
        R = zeros(2); J = zeros(2, 2)
        ss_RJ!(R, J, x, prob)
        # F1 has y[t-1] (∂=-a) and y[t] (∂=1) -> row 1 col 1 = 1 - a = 0.7
        # F1 also has z[t] -> row 1 col 2 = -b = -0.4
        # F2 has z[t-1] (∂=-c) and z[t] (∂=1) -> row 2 col 2 = 1 - c = 0.4
        # F2 has nothing in y -> row 2 col 1 = 0
        @test J[1, 1] ≈ 0.7  atol=1e-12
        @test J[1, 2] ≈ -0.4 atol=1e-12
        @test J[2, 1] ≈ 0.0  atol=1e-12
        @test J[2, 2] ≈ 0.4  atol=1e-12
    end

    # ------------------------------------------------------------------
    # Large model builds and simulates through the solver.
    # ------------------------------------------------------------------

    @testset "synthetic 200-equation model builds + simulates" begin
        # n independent AR(1) recursions sharing one shock:
        #   x_i[t] = α * x_i[t-1] + e[t]
        # Square (n_eq == n_var), all-lag/no-lead. SS is 0 for every var.
        function synthetic_def(n::Int)
            def = ModelDef(:synthetic200)
            IR.add_param!(def, IR.ParamDecl(:α, 0.5; kind = IR.PARAM_SCALAR))
            for i in 1:n
                IR.add_var!(def, IR.VarDecl(Symbol(:x, i)))
            end
            IR.add_shock!(def, IR.ShockDecl(:e))
            for i in 1:n
                xi = Symbol(:x, i)
                res = Expr(:call, :-,
                           Expr(:ref, xi, :t),
                           Expr(:call, :+,
                                Expr(:call, :*, :α, Expr(:ref, xi, :(t - 1))),
                                Expr(:ref, :e, :t)))
                IR.add_equation!(def,
                    IR.EquationAST(res, Set{IR.EquationFlag}(), nothing,
                                   LineNumberNode(i, :synthetic200)))
            end
            return def
        end

        old = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
        try
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "50"   # 200 eqs -> Vector path
            n = 200
            t0 = time()
            compiled = initialize_model(synthetic_def(n))
            build_time = time() - t0
            @test compiled.eqns isa Vector{ModelBaseEcon.Equation}
            @test length(compiled) == n

            T = 12
            plan = StackedTimePlan(compiled, T)
            @test plan.maxlag == 1
            @test plan.maxlead == 0
            @test plan.n_eq == n

            # Initial condition x_i[0] = i; with α=0.5, c=0 the SS is 0,
            # so the closed form is x_i[t] = 0.5^t * i.
            x_init = reshape(Float64[i for i in 1:n], 1, n)
            x_term = zeros(0, n)
            e_full = zeros(T + 1, 1)   # one shock, no impulse

            t1 = time()
            x_full, converged, iters = simulate!(plan; x_init, x_term, e_full)
            sim_time = time() - t1
            @test converged
            @test iters <= 3          # linear system -> 1 Newton step + check
            # 200 x 12 = 2400 cells against an analytic closed form;
            # collapse to one assertion on max |Δ|.
            max_synth_diff = 0.0
            for i in 1:n, t in 1:T
                max_synth_diff = max(max_synth_diff,
                                     abs(x_full[1 + t, i] - 0.5^t * i))
            end
            @test max_synth_diff < 1e-9
            @info "200-eq build+simulate" build_time sim_time iters
        finally
            old === nothing ? delete!(ENV, "RW_MBE_TUPLE_THRESHOLD") :
                              (ENV["RW_MBE_TUPLE_THRESHOLD"] = old)
        end
    end
end # function run_core_solver!
