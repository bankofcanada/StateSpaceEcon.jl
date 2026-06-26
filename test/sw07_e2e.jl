##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# Smets & Wouters (2007) end-to-end.
#
# Transcribed from TutorialsEcon.jl/3.US_SW07/SW07.jl. SW07 is a
# log-linearised DSGE model: 41 variables, 7 shocks, 41 equations, deep
# @link parameter chains. Variables are log-deviations / observables, so
# all equations are linear -> SS solve and stacked-time solve are each a
# single Newton step.
#
# This file builds the model on the ModelBaseEcon surface and runs SS +
# an impulse simulation. The legacy numerical match lives in
# test/sw07_legacy_match.jl.
#
# fctype: the capture uses fctype=fclevel (terminal = SS), matching the
# StateSpaceEcon stacked-time solver's x_term = SS clamp.
# ----------------------------------------------------------------------

"""
Build the SW07 model on the ModelBaseEcon surface. Returns the
frozen `Model`. The @link chain is resolved symbolically at @initialize.
"""
function build_sw07()
    m = ModelDef(:SW07)
    @parameters m begin
        # fixed parameters
        ctou    = 0.025
        clandaw = 1.5
        cg      = 0.18
        curvp   = 10.0
        curvw   = 10.0
        # estimated parameters
        ctrend     = 0.4312
        cgamma     = @link ctrend / 100 + 1
        constebeta = 0.1657
        cbeta      = @link 100 / (constebeta + 100)
        constepinf = 0.7869
        cpie       = @link constepinf / 100 + 1
        constelab  = 0.5509
        calfa      = 0.1901
        csigma     = 1.3808
        cfc        = 1.6064
        cgy        = 0.5187
        csadjcost  = 5.7606
        chabb      = 0.7133
        cprobw     = 0.7061
        csigl      = 1.8383
        cprobp     = 0.6523
        cindw      = 0.5845
        cindp      = 0.2432
        czcap      = 0.5462
        crpi       = 2.0443
        crr        = 0.8103
        cry        = 0.0882
        crdy       = 0.2247
        crhoa      = 0.9577
        crhob      = 0.2194
        crhog      = 0.9767
        crhoqs     = 0.7113
        crhoms     = 0.1479
        crhopinf   = 0.8895
        crhow      = 0.9688
        cmap       = 0.7010
        cmaw       = 0.8503
        # derived from steady state (@link chains)
        clandap  = @link cfc
        cbetabar = @link cbeta * cgamma^(-csigma)
        cr       = @link cpie / (cbeta * cgamma^(-csigma))
        crk      = @link (cbeta^(-1)) * (cgamma^csigma) - (1 - ctou)
        cw       = @link (calfa^calfa * (1 - calfa)^(1 - calfa) /
                          (clandap * crk^calfa))^(1 / (1 - calfa))
        cikbar   = @link (1 - (1 - ctou) / cgamma)
        cik      = @link (1 - (1 - ctou) / cgamma) * cgamma
        clk      = @link ((1 - calfa) / calfa) * (crk / cw)
        cky      = @link cfc * (clk)^(calfa - 1)
        ciy      = @link cik * cky
        ccy      = @link 1 - cg - cik * cky
        crkky    = @link crk * cky
        cwhlc    = @link (1 / clandaw) * (1 - calfa) / calfa * crk * cky / ccy
        cwly     = @link 1 - crk * cky
    end

    @variables m begin
        a; b; c; cf; dc
        dinve; dw; dy; epinfma; ewma
        g; inve; invef; k; kf
        kp; kpf; lab; labf; labobs
        mc; ms; pinf; pinf4; pinfobs
        pk; pkf; qs; r; rk
        rkf; robs; rrf; spinf; sw
        w; wf; y; yf; zcap
        zcapf
    end

    @shocks m begin
        ea; eb; eg; em; epinf; eqs; ew
    end

    @autoexogenize m begin
        labobs  = eg
        robs    = em
        pinfobs = epinf
        dy      = ea
        dc      = eb
        dinve   = eqs
        dw      = ew
    end

    @equations m begin
        # flexible economy
        0 * (1 - calfa) * a[t] + 1 * a[t] = calfa * rkf[t] + (1 - calfa) * (wf[t])
        zcapf[t] = (1 / (czcap / (1 - czcap))) * rkf[t]
        rkf[t] = (wf[t]) + labf[t] - kf[t]
        kf[t] = kpf[t - 1] + zcapf[t]
        "investment Euler equation"
        invef[t] = (1 / (1 + cbetabar * cgamma)) * (invef[t - 1] + cbetabar * cgamma * invef[t + 1] + (1 / (cgamma^2 * csadjcost)) * pkf[t]) + qs[t]
        pkf[t] = -rrf[t] - 0 * b[t] + (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) * b[t] + (crk / (crk + (1 - ctou))) * rkf[t + 1] + ((1 - ctou) / (crk + (1 - ctou))) * pkf[t + 1]
        "consumption Euler equation"
        cf[t] = (chabb / cgamma) / (1 + chabb / cgamma) * cf[t - 1] + (1 / (1 + chabb / cgamma)) * cf[t + 1] + ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) * (labf[t] - labf[t + 1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * (rrf[t] + 0 * b[t]) + b[t]
        "aggregate resource constraint"
        yf[t] = ccy * cf[t] + ciy * invef[t] + g[t] + crkky * zcapf[t]
        "aggregate production function"
        yf[t] = cfc * (calfa * kf[t] + (1 - calfa) * labf[t] + a[t])
        wf[t] = csigl * labf[t] + (1 / (1 - chabb / cgamma)) * cf[t] - (chabb / cgamma) / (1 - chabb / cgamma) * cf[t - 1]
        "accumulation of installed capital"
        kpf[t] = (1 - cikbar) * kpf[t - 1] + (cikbar) * invef[t] + (cikbar) * (cgamma^2 * csadjcost) * qs[t]

        # sticky price - wage economy
        "marginal cost"
        mc[t] = calfa * rk[t] + (1 - calfa) * (w[t]) - 1 * a[t] - 0 * (1 - calfa) * a[t]
        "capital utilization"
        zcap[t] = (1 / (czcap / (1 - czcap))) * rk[t]
        "rental rate of capital"
        rk[t] = w[t] + lab[t] - k[t]
        "Capital installed used one period later in production"
        k[t] = kp[t - 1] + zcap[t]
        "investment Euler equation"
        inve[t] = (1 / (1 + cbetabar * cgamma)) * (inve[t - 1] + cbetabar * cgamma * inve[t + 1] + (1 / (cgamma^2 * csadjcost)) * pk[t]) + qs[t]
        "arbitrage equation for value of capital"
        pk[t] = -r[t] + pinf[t + 1] - 0 * b[t] + (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) * b[t] + (crk / (crk + (1 - ctou))) * rk[t + 1] + ((1 - ctou) / (crk + (1 - ctou))) * pk[t + 1]
        "consumption Euler equation"
        c[t] = (chabb / cgamma) / (1 + chabb / cgamma) * c[t - 1] + (1 / (1 + chabb / cgamma)) * c[t + 1] + ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) * (lab[t] - lab[t + 1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * (r[t] - pinf[t + 1] + 0 * b[t]) + b[t]
        "aggregate resource constraint"
        y[t] = ccy * c[t] + ciy * inve[t] + g[t] + 1 * crkky * zcap[t]
        "aggregate production function"
        y[t] = cfc * (calfa * k[t] + (1 - calfa) * lab[t] + a[t])
        "Phillips Curve"
        pinf[t] = (1 / (1 + cbetabar * cgamma * cindp)) * (cbetabar * cgamma * pinf[t + 1] + cindp * pinf[t - 1] + ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp) / ((cfc - 1) * curvp + 1) * (mc[t])) + spinf[t]
        w[t] = (1 / (1 + cbetabar * cgamma)) * w[t - 1] + (cbetabar * cgamma / (1 + cbetabar * cgamma)) * w[t + 1] + (cindw / (1 + cbetabar * cgamma)) * pinf[t - 1] - (1 + cbetabar * cgamma * cindw) / (1 + cbetabar * cgamma) * pinf[t] + (cbetabar * cgamma) / (1 + cbetabar * cgamma) * pinf[t + 1] + (1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / ((1 + cbetabar * cgamma) * cprobw) * (1 / ((clandaw - 1) * curvw + 1)) * (csigl * lab[t] + (1 / (1 - chabb / cgamma)) * c[t] - ((chabb / cgamma) / (1 - chabb / cgamma)) * c[t - 1] - w[t]) + 1 * sw[t]
        "Monetary Policy Rule"
        r[t] = crpi * (1 - crr) * pinf[t] + cry * (1 - crr) * (y[t] - yf[t]) + crdy * (y[t] - yf[t] - y[t - 1] + yf[t - 1]) + crr * r[t - 1] + ms[t]
        a[t] = crhoa * a[t - 1] + ea[t]
        b[t] = crhob * b[t - 1] + eb[t]
        "exogenous spending (also including net exports)"
        g[t] = crhog * (g[t - 1]) + eg[t] + cgy * ea[t]
        qs[t] = crhoqs * qs[t - 1] + eqs[t]
        ms[t] = crhoms * ms[t - 1] + em[t]
        "cost push shock"
        spinf[t] = crhopinf * spinf[t - 1] + epinfma[t] - cmap * epinfma[t - 1]
        epinfma[t] = epinf[t]
        sw[t] = crhow * sw[t - 1] + ewma[t] - cmaw * ewma[t - 1]
        ewma[t] = ew[t]
        "accumulation of installed capital"
        kp[t] = (1 - cikbar) * kp[t - 1] + cikbar * inve[t] + cikbar * cgamma^2 * csadjcost * qs[t]

        # measurement equations
        dy[t] = y[t] - y[t - 1] + ctrend
        dc[t] = c[t] - c[t - 1] + ctrend
        dinve[t] = inve[t] - inve[t - 1] + ctrend
        dw[t] = w[t] - w[t - 1] + ctrend
        pinfobs[t] = 1 * (pinf[t]) + constepinf
        pinf4[t] = pinf[t] + pinf[t - 1] + pinf[t - 2] + pinf[t - 3]
        robs[t] = 1 * (r[t]) + constebeta
        labobs[t] = lab[t] + constelab
    end

    return @initialize m
end

# Wrapped in a runner (see runtests.jl) so the suite can dispatch this
# self-contained testset on its own thread. `build_sw07` above stays a
# top-level function so sw07_legacy_match.jl can call it.
function run_sw07_e2e!()
@testset "SW07 end-to-end" begin

    compiled = build_sw07()

    @testset "model builds with 41 vars / 7 shocks / 41 eqns" begin
        @test compiled isa CompiledModel
        @test length(compiled.defs.vars) == 41
        @test length(compiled.defs.shocks) == 7
        @test length(compiled) == 41
        # @link chain resolved away: param_layout holds only root params.
        root_names = Set(pr.name for pr in compiled.param_layout)
        @test :ctou in root_names          # a root scalar
        @test !(:cbeta in root_names)      # a linked param
        @test !(:cwhlc in root_names)      # a deep-linked param
    end

    @testset "steady state solves" begin
        prob = SteadyStateProblem(compiled)
        @test prob.n_var == 41
        @test prob.n_eq == 41
        # SW07 SS is mostly zero; the 7 measurement vars sit at constants.
        # A zero start is in the linear basin.
        x_ss, converged, iters = sssolve!(prob;
            x0 = zeros(41), tol = 1e-12, maxiter = 50)
        @test converged
        # Linear model -> converges fast.
        @test iters <= 5
    end

    @testset "impulse simulation runs and stays finite" begin
        prob = SteadyStateProblem(compiled)
        x_ss, conv, _ = sssolve!(prob; x0 = zeros(41), tol = 1e-12)
        @test conv

        sim_T = 40
        plan = StackedTimePlan(compiled, sim_T)
        n = 41
        # SW07 has pinf[t-3] (maxlag=3) and leads (maxlead=1); boundary
        # blocks are the SS row repeated maxlag / maxlead times.
        x_init = repeat(reshape(x_ss, 1, n), plan.maxlag, 1)
        x_term = repeat(reshape(x_ss, 1, n), plan.maxlead, 1)
        e_full = zeros(sim_T + plan.maxlag + plan.maxlead, 7)
        # `em` monetary shock at the first interior period.
        em_idx = plan.shock_index[:em]
        e_full[plan.maxlag + 1, em_idx] = 0.01

        x_guess = repeat(reshape(x_ss, 1, n), sim_T, 1)
        x_full, sconv, iters = simulate!(plan;
            x_init, x_term, e_full, x_guess, tol = 1e-12, maxiter = 80)
        @test sconv
        @test all(isfinite, x_full)
    end
end  # @testset
end  # function run_sw07_e2e!
