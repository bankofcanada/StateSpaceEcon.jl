##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Pardiso dispatch + (gated) numerical-match tests.
#
# CI exercises the dispatch logic only. The Pardiso-backed solve runs
# only when ENV["PARDISO_CI"] == "1"; in default CI the absence of
# Pardiso must produce a clean error message pointing at `using Pardiso`.
# Pardiso.jl is NOT a default test dep (heavyweight, MKL install pain on
# macOS/Windows) - the gated test loads it via `@eval using Pardiso`
# from the user's depot, mirroring the `_HAS_TSE` pattern.
# ----------------------------------------------------------------------

function run_pardiso_dispatch_tests!()
@testset "Pardiso dispatch" begin

    # Tiny AR(1) plan reused across testsets. SS analytically 1/(1-α).
    function _build_ar1_plan(T::Int)
        m = ModelDef(:pardiso_ar1)
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + c + e[t]
        end
        compiled = @initialize m
        return compiled, StackedTimePlan(compiled, T)
    end

    # ------------------------------------------------------------------
    # (1) Default :umfpack path still works (regression guard).
    # ------------------------------------------------------------------
    @testset "default :umfpack path" begin
        _, plan = _build_ar1_plan(6)
        y_ss = 1.0 / (1 - 0.5)
        x_init = reshape([y_ss], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(7, 1); e_full[2, 1] = 1.0
        x_full, converged, _ = simulate!(plan;
            x_init, x_term, e_full, linsolve = :umfpack)
        @test converged
        @test x_full[2, 1] ≈ y_ss + 1.0 atol=1e-10
    end

    # ------------------------------------------------------------------
    # (2) Dispatch error: without Pardiso loaded, :pardiso must raise a
    #     clean ErrorException mentioning `using Pardiso`.
    # ------------------------------------------------------------------
    @testset "dispatch error without Pardiso loaded" begin
        _, plan = _build_ar1_plan(4)
        y_ss = 1.0 / (1 - 0.5)
        x_init = reshape([y_ss], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(5, 1); e_full[2, 1] = 1.0
        if isdefined(Main, :Pardiso)
            # Pardiso is loaded in this session (e.g. PARDISO_CI=1 ran
            # above) - the error path is not testable here.
            @test_skip false
        else
            err = nothing
            try
                simulate!(plan; x_init, x_term, e_full, linsolve = :pardiso)
            catch e
                err = e
            end
            @test err isa ErrorException
            @test occursin("using Pardiso", err.msg)
        end
    end

    # ------------------------------------------------------------------
    # (3) Unknown linsolve symbol.
    # ------------------------------------------------------------------
    @testset "unknown linsolve symbol" begin
        _, plan = _build_ar1_plan(4)
        y_ss = 1.0 / (1 - 0.5)
        x_init = reshape([y_ss], 1, 1)
        x_term = zeros(0, 1)
        e_full = zeros(5, 1); e_full[2, 1] = 1.0
        @test_throws ErrorException simulate!(plan;
            x_init, x_term, e_full, linsolve = :nonsense)
    end

    # ------------------------------------------------------------------
    # (4) Gated numerical-match test - only when PARDISO_CI=1 is set in
    #     the environment AND Pardiso resolves from the user's depot.
    #     Collapsed to a single max-|Δ| assertion per the
    #     test-collapsing convention.
    # ------------------------------------------------------------------
    _HAS_PARDISO = if get(ENV, "PARDISO_CI", "0") == "1"
        try
            @eval using Pardiso
            true
        catch e
            @warn "PARDISO_CI=1 but `using Pardiso` failed; skipping gated tests" exception=e
            false
        end
    else
        false
    end

    if _HAS_PARDISO
        @testset "Pardiso ↔ UMFPACK numerical match (gated)" begin
            _, plan = _build_ar1_plan(20)
            y_ss = 1.0 / (1 - 0.5)
            x_init = reshape([y_ss], 1, 1)
            x_term = zeros(0, 1)
            e_full = zeros(21, 1); e_full[2, 1] = 1.0
            x_u, cu, _ = simulate!(plan;
                x_init, x_term, e_full, linsolve = :umfpack)
            x_p, cp, _ = simulate!(plan;
                x_init, x_term, e_full, linsolve = :pardiso)
            @test cu && cp
            @test maximum(abs.(x_u .- x_p)) < 1e-12
        end

        @testset "Pardiso state release smoke" begin
            # Calling simulate! repeatedly must not leak / crash; the
            # extension's finalizer + try/finally releases the solver
            # handle at the end of each Newton loop.
            _, plan = _build_ar1_plan(4)
            y_ss = 1.0 / (1 - 0.5)
            x_init = reshape([y_ss], 1, 1)
            x_term = zeros(0, 1)
            e_full = zeros(5, 1); e_full[2, 1] = 1.0
            for _ in 1:3
                _, c, _ = simulate!(plan;
                    x_init, x_term, e_full, linsolve = :pardiso)
                @test c
            end
            GC.gc()  # force finalizers to fire; smoke test only.
            @test true
        end
    end

end
end
