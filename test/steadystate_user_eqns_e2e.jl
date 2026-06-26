##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# @steadystate user-supplied SS constraints (StateSpaceEcon side).

function run_steadystate_user_eqns_tests!()
    @testset "@steadystate augmented SS solve" begin

        # ------------------------------------------------------------------
        # (1) Square augmented system. Pure forward-looking: y = β*y[t+1] + c
        #     analytic SS = c/(1-β). Add @steadystate y = c_target; the
        #     auto-equation alone determines y, so the constraint must
        #     either agree or the system is over-determined. Drop the
        #     auto-eq from determinacy by using an under-determined dynamic
        #     model (2 vars, 1 dynamic eq) and one @steadystate constraint
        #     to close it.
        # ------------------------------------------------------------------
        @testset "augments under-determined dynamic model" begin
            m = ModelDef(:g4_under)
            @parameters m begin; α = 0.5; end
            @variables m begin; y; z; end
            # One dynamic eq, two variables -> SS is under-determined
            # without a user constraint. Adding `z = 0.7` closes it.
            @equations m begin
                y[t] = α * y[t-1] + z[t]
            end
            @steadystate m z = 0.7
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            @test prob.n_eq == 1
            @test prob.n_ss_user == 1
            x_ss, conv, _ = sssolve!(prob; x0 = [0.0, 0.0], tol = 1e-12)
            @test conv
            # z_ss = 0.7 by constraint, y_ss = z_ss/(1-α) = 1.4
            @test isapprox(x_ss[prob.var_index[:z]], 0.7; atol = 1e-10)
            @test isapprox(x_ss[prob.var_index[:y]], 1.4; atol = 1e-10)
        end

        # ------------------------------------------------------------------
        # (2) Two user constraints in a block, closing a 3-var / 1-eq
        #     model.
        # ------------------------------------------------------------------
        @testset "two @steadystate constraints in block form" begin
            m = ModelDef(:g4_under3)
            @parameters m begin; α = 0.5; end
            @variables m begin; y; z; w; end
            @equations m begin
                y[t] = α * y[t-1] + z[t] + w[t]
            end
            @steadystate m begin
                z = 0.4
                w = 0.6
            end
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            @test prob.n_eq == 1
            @test prob.n_ss_user == 2
            x_ss, conv, _ = sssolve!(prob; x0 = zeros(3), tol = 1e-12)
            @test conv
            @test isapprox(x_ss[prob.var_index[:z]], 0.4; atol = 1e-10)
            @test isapprox(x_ss[prob.var_index[:w]], 0.6; atol = 1e-10)
            # y_ss = (z+w)/(1-α) = 1.0/0.5 = 2.0
            @test isapprox(x_ss[prob.var_index[:y]], 2.0; atol = 1e-10)
        end

        # ------------------------------------------------------------------
        # (3) Over-determined: square model + one extra constraint.
        #     Pure consistent case - the constraint agrees with the
        #     auto-derived SS, so the least-squares step still hits
        #     residual zero.
        # ------------------------------------------------------------------
        @testset "over-determined consistent: least-squares converges" begin
            m = ModelDef(:g4_over)
            @parameters m begin; α = 0.5; c0 = 1.0; end
            @variables m begin; y; end
            @equations m begin
                y[t] = α * y[t-1] + c0
            end
            # Analytic SS: y = c0/(1-α) = 2.0; user agrees.
            @steadystate m y = 2.0
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            @test prob.n_eq == 1
            @test prob.n_ss_user == 1
            @test prob.n_var == 1
            x_ss, conv, _ = sssolve!(prob; x0 = [0.0], tol = 1e-12)
            @test conv
            @test isapprox(x_ss[1], 2.0; atol = 1e-10)
        end

        # ------------------------------------------------------------------
        # (4) Over-determined inconsistent: user constraint conflicts with
        #     auto-SS. Least-squares step finds the minimiser; converged
        #     == false because residuals can't all hit `tol`. This is the
        #     handoff to diagnose_sstate - `on_failure = :diagnose` returns
        #     a usable `SteadyStateDiagnosis`.
        # ------------------------------------------------------------------
        @testset "over-determined inconsistent: least-squares + on_failure" begin
            m = ModelDef(:g4_over_bad)
            @parameters m begin; α = 0.5; c0 = 1.0; end
            @variables m begin; y; end
            @equations m begin
                y[t] = α * y[t-1] + c0
            end
            # Analytic SS = 2.0; force a conflicting constraint.
            @steadystate m y = 3.0
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            res = sssolve!(prob; x0 = [0.0], tol = 1e-12, maxiter = 30,
                            on_failure = :diagnose)
            @test length(res) == 4   # not converged -> diagnosis appended
            @test res[2] == false
            d = res[4]
            @test d isa SteadyStateDiagnosis
            @test length(d.residuals) == 2
            # Final iterate sits between 2.0 (auto-SS) and 3.0 (constraint).
            x_final = res[1]
            @test 2.0 - 1e-6 <= x_final[1] <= 3.0 + 1e-6
        end

        # ------------------------------------------------------------------
        # (5) Constraint uses parameters: `c = c_target` where c_target is
        #     a parameter.
        # ------------------------------------------------------------------
        @testset "@steadystate residual references model parameters" begin
            m = ModelDef(:g4_params)
            @parameters m begin; α = 0.5; c_target = 1.7; end
            @variables m begin; y; z; end
            @equations m begin
                y[t] = α * y[t-1] + z[t]
            end
            @steadystate m z = c_target
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            x_ss, conv, _ = sssolve!(prob; x0 = zeros(2), tol = 1e-12)
            @test conv
            @test isapprox(x_ss[prob.var_index[:z]], 1.7; atol = 1e-10)
            @test isapprox(x_ss[prob.var_index[:y]], 1.7 / 0.5; atol = 1e-10)
        end

        # ------------------------------------------------------------------
        # (6) @delete reverts to the auto-derived SS.
        # ------------------------------------------------------------------
        @testset "@delete restores baseline SS shape" begin
            m = ModelDef(:g4_del_e2e)
            @parameters m begin; α = 0.5; c0 = 1.0; end
            @variables m begin; y; end
            @equations m begin
                y[t] = α * y[t-1] + c0
            end
            @steadystate m y = 99.0   # would conflict with auto-SS
            @steadystate m @delete _SSEQ1
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            @test prob.n_ss_user == 0
            x_ss, conv, _ = sssolve!(prob; x0 = [0.0], tol = 1e-12)
            @test conv
            @test isapprox(x_ss[1], 2.0; atol = 1e-10)
        end
    end
end
