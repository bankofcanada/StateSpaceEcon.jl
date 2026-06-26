"""
    StateSpaceEcon

A package for Macroeconomic modelling.

"""
module StateSpaceEcon

using ModelBaseEcon
using LinearAlgebra: I
using PrecompileTools: @setup_workload, @compile_workload

include("SteadyStateSolver.jl")
include("StackedTimeSolver.jl")
include("Plans.jl")
include("stackedtime/shockdecomp.jl")
include("Kalman.jl")
include("FirstOrderSolver.jl")
include("DFMSolver.jl")
include("stackedtime/stoch_simulate.jl")
include("compat.jl")

using .SteadyStateSolver
using .StackedTimeSolver
using .Plans
using .ShockDecomp
using .Kalman
using .FirstOrderSolver
using .DFMSolver
using .StochSimulate
using .Compat

export SteadyStateProblem, sssolve!, ss_residual!, ss_RJ!
export SteadyStateDiagnosis, diagnose_sstate
export StackedTimePlan, simulate!, stacked_residual!, stacked_RJ!
export SimData, SimPlan, exogenize!, endogenize!, autoexogenize_plan!,
       plan_simulate!, sim_range_length, load_longbase, load_longbase_mvts,
       level_value, set_level!
# Shock decomposition.
export ShockDecompResult, shock_decomp
# Kalman filter / smoother.
export Kalman
export kf_length_x, kf_length_y, kf_is_linear, kf_linear_model
export KFLinearModel, KFilter
export KFDataFilter, KFDataFilterEx, KFDataSmoother, KFDataSmootherEx
export kf_filter, kf_smoother, kf_filter!, kf_smoother!
export @kfd_get, @kfd_set!, @kfd_view
# First-order rational-expectations (QZ) solver.
export FirstOrderModel, first_order_solve, first_order_simulate
export FirstOrderShockDecompResult, first_order_shockdecomp
# Dynamic factor model (DFM) EM solver.
export DFMSolver, EMestimate!
# Stochastic (multi-path) simulation.
export stoch_simulate, StochResult, SimFailed, isfailed
# Compatibility aliases.
export Plan, simulate, solve!, use_pardiso, use_umfpack
export FinalCondition, FCNone, FCGiven, FCMatchSSLevel, fclevel, setfc!

# ---------------------------------------------------------------------------
# Precompile workload (DFM EM)
#
# The first DFM the suite estimates pays ~19s of one-time JIT, almost all of it
# inside `EMestimate!` (kf_filter/kf_smoother bridge + the EM-step linear
# algebra over ComponentArrays-typed params) - measured 19.0s/99.96%
# compilation cold, 0.001s warm. That cost lands entirely on the first DFM
# testset. Running a 2-iteration EM on a trivial DFM here moves that compile
# into the cached package image so the test suite starts warm. The real
# models' EM iterations remain genuine compute and are unaffected. We also
# exercise the standalone kf_filter/kf_smoother convenience methods used by
# the filter/smoother match tests.
@setup_workload begin
    @compile_workload begin
        # Equation-model core: build -> steady-state solve -> stacked-time
        # simulate on a tiny AR(1). These are the first paths every non-DFM
        # testset hits, and they JIT cold the first time in an SSE process -
        # ~4.4s @initialize, ~1.0s sssolve!, ~1.9s StackedTimePlan+simulate!
        # (about 7s cold, 0.085s warm). Without this the cost lands on the
        # first sssolve testset and first simulate testset.
        em = ModelDef(:precompile_ar1)
        @parameters em begin
            α = 0.5
            c = 1.0
        end
        @variables em begin
            y
        end
        @equations em begin
            y[t] = α * y[t-1] + c
        end
        cm = @initialize em
        prob = SteadyStateProblem(cm)
        sssolve!(prob; x0=[0.5], tol=1e-12)
        plan = StackedTimePlan(cm, 10)
        simulate!(plan;
            x_init=reshape([0.0], 1, 1),
            x_term=zeros(0, 1),
            e_full=zeros(11, 0),
        )

        # First-order (QZ/Schur) solver: a distinct ~1.6s cold compile (the
        # generalized-Schur + decision-rule LAPACK specialization) that the
        # first-order testset otherwise pays at first use. A tiny
        # forward-looking model with one shock exercises the same solve path.
        fm = ModelDef(:precompile_fo)
        @parameters fm begin
            rho = 0.6
        end
        @variables fm begin
            x
        end
        @shocks fm begin
            x_shk
        end
        @autoexogenize fm begin
            x = x_shk
        end
        @equations fm begin
            x[t+1] = rho * x[t] + x_shk[t]
        end
        fcm = @initialize fm
        fprob = SteadyStateProblem(fcm)
        fx_ss, _, _ = sssolve!(fprob; x0=zeros(fprob.n_var), tol=1e-12)
        first_order_solve(fcm, fx_ss)

        # Stochastic simulation. Compiles `stoch_simulate` ->
        # `SimPlan` / `plan_simulate!` on a model WITH a lead, so the maxlead
        # terminal-row + per-period sub-window + copy-in/out paths land in the
        # cached image. `fm` (x[t+1] = rho*x[t] + x_shk[t]) has exactly a lead
        # and a shock. This covers the stoch/plan_simulate machinery only; the
        # remaining cold cost of the stochastic testset is dominated by genuine
        # model-BUILD codegen, which is first-build work, not avoidable JIT.
        sbase = SimData(fcm, 1, 6; maxlead=1)
        sshk = zeros(1, 1, 2)
        sshk[1, 1, 1] = 0.01
        stoch_simulate(fcm, sbase, sshk; shock_start=2, tol=1e-9, maxiter=10)

        m = DFM(:precompile_dfm_em)
        add_observed!(m, :a, :b)
        add_components!(m, F=CommonComponents("F"))
        map_loadings!(m, (:a, :b) => :F)
        add_shocks!(m, :a, :b)
        initialize_dfm!(m)

        # deterministic tiny dataset (no Random in precompile)
        nt = 30
        Y = Matrix{Float64}(undef, nt, 2)
        for t in 1:nt
            Y[t, 1] = sinpi(t / 7)
            Y[t, 2] = cospi(t / 5)
        end

        fill!(m.params, NaN)
        EMestimate!(m, copy(Y); verbose=false, strict=false, maxiter=2)

        # standalone Kalman convenience paths (filter/smoother match tests)
        nx = nstates_with_lags(m)
        x0 = zeros(nx)
        Px0 = Matrix{Float64}(I(nx))
        kf_filter(Y, x0, Px0, m)
        kf_smoother(Y, x0, Px0, m)
    end
end

end # module
