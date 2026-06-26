##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# First-order (QZ) rational-expectations solver, legacy numerical match.
#
# Reference data is captured from the legacy StateSpaceEcon.FirstOrderSolver
# by `legacy_reference/capture_firstorder.jl` (run separately against the
# legacy clones) and committed to `legacy_reference/firstorder_reference.json`.
# This test rebuilds the same three sim_fo.jl models on the ModelBaseEcon DSL,
# runs `first_order_solve` / `first_order_simulate`, and asserts a cell-by-cell
# match at atol=1e-10.
#
# Models (start from legacy sim_fo.jl):
#   M  - scalar forward-looking: x[t+1] = rho*x[t] + x_shk[t]  (nbck=0, nfwd=1)
#   R  - same in @log space:     log(x[t]) = rho*log(x[t-1]) + x_shk[t]
#   E2 - 3-var/3-shock New-Keynesian DSGE: leads AND lags (nbck=3, nfwd=2,
#        mixed fwd/bck cross-links, a complex-conjugate eigenvalue pair).
#
# Per the per-cell-collapse convention the many-cell comparisons collapse to
# one `max|Δ|` assertion each.
#
# CTarget invariant: the first-order solve runs on assembled matrices only -
# ModelBaseEcon kernels are untouched.
# ----------------------------------------------------------------------

using JSON

# ----- model builders (transcribed from legacy sim_fo.jl / E2.jl) -----------

function build_fo_M(rho::Float64 = 0.6)
    m = ModelDef(:M)
    @parameters m begin; rho = rho; end
    @variables m begin; x; end
    @shocks m begin; x_shk; end
    @autoexogenize m begin; x = x_shk; end
    @equations m begin
        x[t+1] = rho * x[t] + x_shk[t]
    end
    return @initialize m
end

function build_fo_R()
    m = ModelDef(:R)
    @parameters m begin; rho = 0.6; end
    @variables m begin; @log x; end
    @shocks m begin; x_shk; end
    @autoexogenize m begin; x = x_shk; end
    @equations m begin
        log(x[t]) = rho * log(x[t-1]) + x_shk[t]
    end
    return @initialize m
end

function build_fo_E2()
    m = ModelDef(:E2)
    @parameters m begin
        cp = [0.5, 0.02]
        cr = [0.75, 1.5, 0.5]
        cy = [0.5, -0.02]
    end
    @variables m begin; pinf; rate; ygap; end
    @shocks m begin; pinf_shk; rate_shk; ygap_shk; end
    @autoexogenize m begin
        pinf = pinf_shk; rate = rate_shk; ygap = ygap_shk
    end
    @equations m begin
        pinf[t] = cp[1] * pinf[t-1] + (0.98 - cp[1]) * pinf[t+1] + cp[2] * ygap[t] + pinf_shk[t]
        rate[t] = cr[1] * rate[t-1] + (1 - cr[1]) * (cr[2] * pinf[t] + cr[3] * ygap[t]) + rate_shk[t]
        ygap[t] = cy[1] * ygap[t-1] + (0.98 - cy[1]) * ygap[t+1] + cy[2] * (rate[t] - pinf[t+1]) + ygap_shk[t]
    end
    return @initialize m
end

function run_firstorder_match_tests!()
@testset "first-order (QZ) solver" begin

    # ------------------------------------------------------------------
    # Internal consistency (no reference data needed): M solves, the
    # decision rule reproduces the published scalar recursion.
    # ------------------------------------------------------------------
    @testset "M — solve + internal invariants" begin
        m = build_fo_M()
        prob = SteadyStateProblem(m)
        x_ss, conv, _ = sssolve!(prob; x0 = zeros(prob.n_var), tol = 1e-12)
        @test conv
        fom = first_order_solve(m, x_ss)
        @test fom.vm.nbck == 0
        @test fom.vm.nfwd == 1
        @test fom.vm.nex == 1
        # the single generalized eigenvalue is -1/rho = -1.6667 (unstable)
        λ = complex(fom.qz.α_re[1], fom.qz.α_im[1]) / fom.qz.β[1]
        @test real(λ) ≈ -1/0.6 atol=1e-10
    end

    # ------------------------------------------------------------------
    # First-order shock decomposition invariants (linear model): the
    # per-source contributions sum to shocked - control bit-identically,
    # and the :nonlinear column is ~0. (No legacy capture for the fo
    # shockdecomp; the invariant is the correctness check.)
    # ------------------------------------------------------------------
    @testset "E2 — shock-decomp invariants" begin
        m = build_fo_E2()
        prob = SteadyStateProblem(m)
        x_ss, _, _ = sssolve!(prob; x0 = zeros(prob.n_var))
        fom = first_order_solve(m, x_ss)
        T = 20
        ncol = prob.n_var + length(m.defs.shocks)
        nrows = fom.maxlag + T + fom.maxlead
        control = zeros(nrows, ncol)
        shocked = zeros(nrows, ncol)
        shocked[fom.maxlag + 1, prob.n_var + 1] = 1.0   # pinf_shk impulse
        shocked[fom.maxlag + 2, prob.n_var + 2] = 0.5   # rate_shk impulse
        shocked[1, 1] = 0.2                              # nonzero init on pinf
        r = first_order_shockdecomp(fom, shocked, control)
        @test r.source_names == [:init, :pinf_shk, :rate_shk, :ygap_shk, :nonlinear]
        sum_err = 0.0
        nl_max = 0.0
        for v in m.defs.vars
            gi = fom.vm.vi[v.name]
            M = r.contrib[v.name]
            for row in 1:nrows
                total = r.shocked[row, gi] - r.control[row, gi]
                sum_err = max(sum_err, abs(sum(M[row, :]) - total))
                nl_max = max(nl_max, abs(M[row, end]))
            end
        end
        @test sum_err < 1e-10        # contributions add up exactly
        @test nl_max < 1e-10         # linear model => no nonlinear residual
    end

    # ------------------------------------------------------------------
    # Fuzz guard: a tiny parameter perturbation
    # must not flip the Blanchard-Kahn stable/unstable counts.
    # ------------------------------------------------------------------
    @testset "M — QZ count stability under perturbation" begin
        m = build_fo_M()
        prob = SteadyStateProblem(m)
        x_ss, _, _ = sssolve!(prob; x0 = zeros(prob.n_var))
        fom0 = first_order_solve(m, x_ss)
        m2 = build_fo_M(0.6 + 1e-8)
        prob2 = SteadyStateProblem(m2)
        x_ss2, _, _ = sssolve!(prob2; x0 = zeros(prob2.n_var))
        fom2 = first_order_solve(m2, x_ss2)
        @test (fom2.vm.nbck, fom2.vm.nfwd) == (fom0.vm.nbck, fom0.vm.nfwd)
    end

    # ------------------------------------------------------------------
    # Numerical match against the captured legacy reference.
    # ------------------------------------------------------------------
    ref_path = abspath(joinpath(@__DIR__, "..", "..", "legacy_reference",
                                "firstorder_reference.json"))
    if !isfile(ref_path)
        @info "skipping first-order legacy match - capture_firstorder.jl not run"
        @test_skip "firstorder_reference.json not present"
    else
        ref = JSON.parsefile(ref_path)
        atol = Float64(ref["atol"])

        # reconstruct a column-major-flattened matrix from the JSON dict
        unmat(d) = reshape(Float64[Float64(x) for x in d["data"]], d["rows"], d["cols"])

        # (builder, ref-key, columns (1-based) that are @log in solver space)
        cases = [(build_fo_M, "M", Int[]),
                 (build_fo_R, "R", Int[1]),
                 (build_fo_E2, "E2", Int[])]

        for (build, key, log_cols) in cases
            @testset "$key — legacy match" begin
                refm = ref[key]
                m = build()
                prob = SteadyStateProblem(m)
                x_ss, conv, _ = sssolve!(prob; x0 = zeros(prob.n_var),
                                         tol = 1e-12, maxiter = 100)
                @test conv
                fom = first_order_solve(m, x_ss)

                # Blanchard-Kahn counts (exact integers)
                @test fom.vm.nbck == refm["nbck"]
                @test fom.vm.nfwd == refm["nfwd"]
                @test fom.vm.nex  == refm["nex"]

                # eigenvalues: compare magnitudes, sorted, to neutralize the
                # ordering of equal-magnitude (complex-conjugate) pairs.
                eig = sort([abs(fom.qz.β[i] == 0 ? Inf :
                              complex(fom.qz.α_re[i], fom.qz.α_im[i]) / fom.qz.β[i])
                            for i in 1:fom.qz.N])
                refeig = sort(abs.(complex.(Float64.(refm["eigvals_re"]),
                                            Float64.(refm["eigvals_im"]))))
                @test maximum(abs.(eig .- refeig)) < atol

                # decision-rule matrices (collapsed per matrix)
                refMAT = unmat(refm["MAT"])
                @test maximum(abs.(fom.MAT .- refMAT)) < atol
                refRbZ = unmat(refm["RbyZbb"])
                if length(refRbZ) > 0
                    @test maximum(abs.(fom.RbyZbb .- refRbZ)) < atol
                else
                    @test length(fom.RbyZbb) == 0
                end

                # 40q simulation under a unit shock on the first shock at the
                # first simulation period (matches the capture protocol).
                T = refm["sim_T"]
                ncol = prob.n_var + length(m.defs.shocks)
                nrows = fom.maxlag + T + fom.maxlead
                vals = zeros(nrows, ncol)
                vals[fom.maxlag + 1, prob.n_var + 1] = 1.0
                out = first_order_simulate(fom, vals)
                # legacy sim is in levels - exp the @log columns
                for c in log_cols
                    out[:, c] = exp.(out[:, c])
                end
                refsim = unmat(refm["sim"])
                @test maximum(abs.(out .- refsim)) < atol
            end
        end
    end

end # @testset
end # function run_firstorder_match_tests!
