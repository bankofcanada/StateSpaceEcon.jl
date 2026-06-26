##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# diagnose_sstate tests.

function run_diagnose_sstate_tests!()
    @testset "diagnose_sstate" begin

        # ------------------------------------------------------------------
        # (1) Clean SS solve - residuals tiny, no near-singular rows.
        # ------------------------------------------------------------------
        @testset "clean SS: residuals < 1e-12 at solution" begin
            m = ModelDef(:clean)
            @parameters m begin; α = 0.5; c = 1.0; end
            @variables m begin; y; end
            @equations m begin
                y[t] = α * y[t-1] + c
            end
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            x_ss, conv, _ = sssolve!(prob; x0 = [0.0])
            @test conv
            d = diagnose_sstate(prob, x_ss)
            @test d isa SteadyStateDiagnosis
            @test maximum(abs, d.residuals) < 1e-12
            @test isempty(d.near_singular_rows)
            @test isfinite(d.jacobian_cond)
            @test d.jacobian_cond > 0
        end

        # ------------------------------------------------------------------
        # (2) Ill-conditioned 3-variable SS, evaluated at a wrong point -
        #     the equation with the largest residual is correctly flagged.
        #
        # Variables: y, z, w. Params chosen so the SS is (1, 2, 3).
        #   y[t] = y[t-1]                       -> R1 = 0  always
        #   z[t] = 0.5*z[t-1] + 1               -> R2 = 0 at z=2
        #   w[t] = 0.1*w[t-1] + 2.7             -> R3 = 0 at w=3
        # At x=(10, 10, 10):
        #   R1 = 0
        #   R2 = 10 - 0.5*10 - 1   = 4
        #   R3 = 10 - 0.1*10 - 2.7 = 6.3   <- worst residual
        # ------------------------------------------------------------------
        @testset "ill-conditioned: worst_residual_eqn is correct" begin
            m = ModelDef(:bad3)
            @parameters m begin
                a1 = 1.0; a2 = 0.5; c2 = 1.0; a3 = 0.1; c3 = 2.7
            end
            @variables m begin; y; z; w; end
            @equations m begin
                y[t] = a1 * y[t-1]
                z[t] = a2 * z[t-1] + c2
                w[t] = a3 * w[t-1] + c3
            end
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            x_attempt = [10.0, 10.0, 10.0]
            d = diagnose_sstate(prob, x_attempt)
            @test d.worst_residual_eqn == 3
            @test isapprox(d.residuals[2], 4.0;  atol = 1e-12)
            @test isapprox(d.residuals[3], 6.3;  atol = 1e-12)
            # Row 1 of J at SS: equation y = a1*y_{-1} collapses to
            # (1 - a1) = 0 for y, zero elsewhere -> near-singular row.
            @test 1 in d.near_singular_rows
            @test 2 ∉ d.near_singular_rows
            @test 3 ∉ d.near_singular_rows
        end

        # ------------------------------------------------------------------
        # (3) Default x_attempt path + show round-trip.
        # ------------------------------------------------------------------
        @testset "default x_attempt and show" begin
            m = ModelDef(:rbcish)
            @parameters m begin; α = 0.6; c = 1.0; end
            @variables m begin; y; end
            @equations m begin
                y[t] = α * y[t-1] + c
            end
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)
            d = diagnose_sstate(prob)  # defaults to ones(n_var)
            @test length(d.residuals) == prob.n_eq
            @test length(d.slopes) == prob.n_eq
            @test d.worst_residual_eqn == 1
            s = sprint(show, d)
            @test occursin("SteadyStateDiagnosis", s)
            @test occursin("worst_residual_eqn", s)
        end

        # ------------------------------------------------------------------
        # (4) sssolve! on_failure semantics.
        #
        #   y[t]^2 + 1 = 0  has no real solution -> Newton can't converge.
        # ------------------------------------------------------------------
        @testset "sssolve! on_failure dispatch" begin
            m = ModelDef(:nosol)
            @variables m begin; y; end
            @equations m begin
                y[t]^2 + 1 = 0
            end
            compiled = @initialize m
            prob = SteadyStateProblem(compiled)

            # :error path - current behaviour, 3-tuple.
            res = sssolve!(prob; x0 = [1.0], maxiter = 5, on_failure = :error)
            @test length(res) == 3
            @test res[2] == false  # not converged

            # :diagnose path - 4-tuple, last element is a diagnosis.
            res2 = sssolve!(prob; x0 = [1.0], maxiter = 5, on_failure = :diagnose)
            @test length(res2) == 4
            @test res2[2] == false
            @test res2[4] isa SteadyStateDiagnosis
            @test res2[4].worst_residual_eqn == 1

            # :warn path - 3-tuple plus a @warn.
            res3 = (@test_logs (:warn, r"sssolve!") sssolve!(prob;
                        x0 = [1.0], maxiter = 5, on_failure = :warn))
            @test length(res3) == 3
            @test res3[2] == false

            # Invalid symbol rejected.
            @test_throws ErrorException sssolve!(prob; x0 = [1.0], on_failure = :bogus)

            # On successful convergence, on_failure is a no-op:
            mC = ModelDef(:conv)
            @parameters mC begin; α = 0.5; c = 1.0; end
            @variables mC begin; y; end
            @equations mC begin
                y[t] = α * y[t-1] + c
            end
            cC = @initialize mC
            pC = SteadyStateProblem(cC)
            r4 = sssolve!(pC; x0 = [0.0], on_failure = :diagnose)
            @test length(r4) == 3   # converged -> no diagnosis appended
            @test r4[2] == true
        end

    end
end
