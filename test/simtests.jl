##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

using LinearAlgebra

@testset "E1.simple" begin
    m = E1.model
    m.α = 0.5
    m.β = 1.0 - m.α
    for T = 6:10
        p = Plan(m, 2:T - 1)
        data = zerodata(m, p)
        data[1, :] = [1 0]   # initial condition
        data[end, :] = [5 0] # final condition
        sim00 = simulate(m, p, data)
        exp00 = hcat(1.0:(5.0 - 1.0) / (T - 1):5.0, zeros(T))
        # @info "T=$T" sim00 exp00 data
        @test sim00 ≈ exp00
        # solution piecewise linear y - one line before and another line after the shock
        data1 = copy(data)
        shk = .1
        y2val = ((T - 2) * (1 + 2 * shk) + 5.0) / (T - 1)
        data1[2U,:y_shk] = shk  # shock at time 2
        sim01 = simulate(m, p, data1)
        exp01 = vcat([1.0 y2val:(5.0 - y2val) / (T - 2):5.0...], [0 shk zeros(T - 2)...])'
        @test sim01 ≈ exp01
        # exogenous-endogenous swap
        # - replicate the solution above backing out the shock that produces it
        data2 = copy(data)
        data2[2U, :y] = sim01[2U, :y]
        p2 = deepcopy(p)
        exogenize!(p2, :y, 2)
        endogenize!(p2, :y_shk, 2)
        sim02 = simulate(m, p2, data2)
        exp02 = sim01
        @test sim02 ≈ exp02
    end
    # test_simulation(E1.model, "data/M1_TestMatrix.csv")
end

using DelimitedFiles

function test_simulation(m, path)
    nvars = ModelBaseEcon.nvariables(m)
    nshks = ModelBaseEcon.nshocks(m)
    # 
    data_chk = readdlm(path, ',', Float64)
    IDX = size(data_chk, 1)
    plan = Plan(m, 1 + m.maxlag:IDX - m.maxlead)
    # 
    p01 = deepcopy(plan)
    autoexogenize!(p01, m, m.maxlag + 1:IDX - m.maxlead)
    data01 = zeroarray(m, p01)
    data01[1:m.maxlag,:] = data_chk[1:m.maxlag,:]
    data01[end - m.maxlead + 1:end,:] = data_chk[end - m.maxlead + 1:end,:]
    data01[m.maxlag + 1:end - m.maxlead,1:nvars] = data_chk[m.maxlag + 1:end - m.maxlead,1:nvars]
    res01 = simulate(m, p01, data01)
    @test res01 ≈ data_chk
    p02 = deepcopy(plan)
    data02 = zeroarray(m, p02)
    data02[1:m.maxlag,:] = data_chk[1:m.maxlag,:]
    data02[end - m.maxlead + 1:end,:] = data_chk[end - m.maxlead + 1:end,:]
    data02[m.maxlag + 1:end - m.maxlead,nvars .+ (1:nshks)] = data_chk[m.maxlag + 1:end - m.maxlead,nvars .+ (1:nshks)]
    res02 = simulate(m, p02, data02)
    @test res02 ≈ data_chk
end

@testset "E1.sim" begin
    test_simulation(E1.model, "data/M1_TestData.csv")
end

@testset "E2.sim" begin
    test_simulation(E2.model, "data/M2_TestData.csv")
end

@testset "E3.sim" begin
    test_simulation(E3.model, "data/M3_TestData.csv")
end

@testset "E6.sim" begin
    test_simulation(E6.model, "data/M6_TestData.csv")
end

#############################################################
# linearization tests

@testset "linearize" begin
    m3 = deepcopy(E3.model)
    clear_sstate!(m3)
    @test_throws ModelBaseEcon.LinearizationError linearize!(m3)

    sssolve!(m3)
    @test issssolved(m3)

    test_simulation(m3, "data/M3_TestData.csv")

    m3nl = deepcopy(E3nl.model)
    m3nl.sstate.values .= m3.sstate.values
    m3nl.sstate.mask .= m3.sstate.mask
    @test issssolved(m3nl)

    # test_simulation(m3nl, "data/M3_TestMatrix.csv")

    with_linearized(m3nl) do foo
        test_simulation(foo, "data/M3_TestData.csv")
    end

    @test isa(m3nl.evaldata, ModelBaseEcon.ModelEvaluationData)

    linearize!(m3)
    @test isa(m3.evaldata, ModelBaseEcon.LinearizedModelEvaluationData)
    test_simulation(m3, "data/M3_TestData.csv")

    m7 = deepcopy(E7.model)
    if !issssolved(m7)
        empty!(m7.sstate.constraints)
        @steadystate m7 linv = lc - 7;
        @steadystate m7 lc = 14;
        clear_sstate!(m7)
        sssolve!(m7);
    end
    # the aux variables are present
    @test_throws ModelBaseEcon.LinearizationError linearize!(m7)

    clear_sstate!(E7A.model)
    sssolve!(E7A.model)
    # non-zeros linear growth in the steady state
    @test_throws ModelBaseEcon.LinearizationError linearize!(E7A.model)

end


#############################################################
# Tests of unanticipated shocks

@testset "E1.unant" begin
    m = E1.model
    m.α = m.β = 0.5
    p = Plan(m, 1:3);
    data = zeroarray(m, p);
    data[1,1] = 1.0;
    data[end,1] = 5.0;
    data[3,2] = 0.1;
    res_u = simulate(m, p, data; anticipate=false)
    true_u = Float64[1 0; 2 0; 47 / 15 0.1; 61 / 15 0; 5 0]
    @test res_u ≈ true_u atol = 1e-12
end

@testset "E2.unant" begin
    m = E2.model
    nvars = length(m.variables)
    nshks = length(m.shocks)
    var_inds = 1:nvars
    shk_inds = nvars .+ (1:nshks)
    # Steady state
    m.sstate.values .= 0;
    m.sstate.mask   .= true;
    let atrue = readdlm("./data/M2_Ant.csv", ',', Float64),
        utrue = readdlm("./data/M2_Unant.csv", ',', Float64)
        # Set simulation ranges and plan
        IDX = size(atrue, 1)
        init = 1:m.maxlag
        sim = 1 + m.maxlag:IDX - m.maxlead
        term = IDX .+ (1 - m.maxlead:0)
        p = Plan(m, sim)
        # Make up the exogenous data for the experiment
        let adata = zeroarray(m, p),
            udata = zeroarray(m, p)
            adata[sim,shk_inds] .= atrue[sim,shk_inds]
            udata[sim,shk_inds] .= utrue[sim,shk_inds]
            adata[term,:] .= atrue[term,:]
            udata[term,:] .= utrue[term,:]
            # Run the simulations and test
            ares = simulate(m, p, adata)
            @test ares ≈ atrue atol = m.options.tol
            ures = simulate(m, p, udata; anticipate=false)
            @test ures ≈ utrue atol = m.options.tol
        end
        let adata = zeroarray(m, p),
            udata = zeroarray(m, p)
            autoexogenize!(p, m, sim)
            adata[sim,var_inds] .= atrue[sim,var_inds]
            udata[sim,var_inds] .= utrue[sim,var_inds]
            adata[term,:] .= atrue[term,:]
            udata[term,:] .= utrue[term,:]
            # Run the simulations and test
            ares = simulate(m, p, adata)
            @test ares ≈ atrue atol = m.options.tol
            ures = simulate(m, p, udata; anticipate=false)
            @test ures ≈ utrue atol = m.options.tol
        end
    end
    for s in (1, 4)
        atrue = readdlm("./data/M2_t$(s)_Ant.csv", ',', Float64)
        utrue = readdlm("./data/M2_t$(s)_Unant.csv", ',', Float64)
        # Set simulation ranges and plan
        IDX = size(atrue, 1)
        init = 1:m.maxlag
        sim = 1 + m.maxlag:IDX - m.maxlead
        term = IDX .+ (1 - m.maxlead:0)
        p = Plan(m, sim)
        # Make up the exogenous data for the experiment
        adata = zeroarray(m, p);
        udata = zeroarray(m, p);
        ygap_ind = indexin([:ygap], m.variables)[1]
        adata[sim[1:s], ygap_ind] .= -0.10
        udata[sim[1:s], ygap_ind] .= -0.10
        adata[term,:] .= atrue[term,:]
        udata[term,:] .= utrue[term,:]
        exogenize!(p, :ygap, sim[1:s])
        endogenize!(p, :ygap_shk, sim[1:s])
        # Run the simulations and test
        ares = simulate(m, p, adata)
        @test ares ≈ atrue atol = m.options.tol
        ures = simulate(m, p, udata; anticipate=false)
        @test ures ≈ utrue atol = m.options.tol
    end
end

@testset "E3.unant" begin
    m = E3.model
    nvars = length(m.variables)
    nshks = length(m.shocks)
    var_inds = 1:nvars
    shk_inds = nvars .+ (1:nshks)
    # Steady state
    m.sstate.values .= 0;
    m.sstate.mask   .= true;
    let atrue = readdlm("./data/M3_Ant.csv", ',', Float64),
        utrue = readdlm("./data/M3_Unant.csv", ',', Float64)
        # Set simulation ranges and plan
        IDX = size(atrue, 1)
        init = 1:m.maxlag
        sim = 1 + m.maxlag:IDX - m.maxlead
        term = IDX .+ (1 - m.maxlead:0)
        p = Plan(m, sim)
        # Make up the exogenous data for the experiment
        let adata = zeroarray(m, p),
            udata = zeroarray(m, p)
            adata[sim,shk_inds] .= atrue[sim,shk_inds]
            udata[sim,shk_inds] .= utrue[sim,shk_inds]
            adata[term,:] .= atrue[term,:]
            udata[term,:] .= utrue[term,:]
            # Run the simulations and test
            ares = simulate(m, p, adata)
            @test ares ≈ atrue atol = m.options.tol
            ures = simulate(m, p, udata; anticipate=false)
            @test ures ≈ utrue atol = m.options.tol
        end
        let adata = zeroarray(m, p),
            udata = zeroarray(m, p)
            autoexogenize!(p, m, sim)
            adata[sim,var_inds] .= atrue[sim,var_inds]
            udata[sim,var_inds] .= utrue[sim,var_inds]
            adata[term,:] .= atrue[term,:]
            udata[term,:] .= utrue[term,:]
            # Run the simulations and test
            ares = simulate(m, p, adata)
            @test ares ≈ atrue atol = m.options.tol
            ures = simulate(m, p, udata; anticipate=false)
            @test ures ≈ utrue atol = m.options.tol
        end
        # tests with different final conditions and expectation_horizon values.
        for (fc, eh) = Iterators.product((fcgiven, fclevel, fcslope, fcnatural), ( nothing, 30, 50, 100))
            # Make up the exogenous data for the experiment
            p = Plan(m, sim)
            udata = zeroarray(m, p)
            udata[sim,shk_inds] .= utrue[sim,shk_inds]
            udata[term,:] .= utrue[term,:]
            # Run the simulations and test
            ures = simulate(m, p, udata; anticipate=false, expectation_horizon=eh, fctype=fc)
            @test ures ≈ utrue atol = m.options.tol  rtol = maximum(abs, utrue[term,:])
            # Repeat with autoexogenize
            p = Plan(m, sim)
            udata = zeroarray(m, p)
            autoexogenize!(p, m, sim)
            udata[sim,var_inds] .= utrue[sim,var_inds]
            udata[term,:] .= utrue[term,:]
            # Run the simulations and test
            ures = simulate(m, p, udata; anticipate=false, expectation_horizon=eh, fctype=fc)
            @test ures ≈ utrue atol = m.options.tol rtol = maximum(abs, utrue[term,:])
        end
    end
    for s in (1, 4)
        atrue = readdlm("./data/M3_t$(s)_Ant.csv", ',', Float64)
        utrue = readdlm("./data/M3_t$(s)_Unant.csv", ',', Float64)
        # Set simulation ranges and plan
        IDX = size(atrue, 1)
        init = 1:m.maxlag
        sim = 1 + m.maxlag:IDX - m.maxlead
        term = IDX .+ (1 - m.maxlead:0)
        p = Plan(m, sim)
        # Make up the exogenous data for the experiment
        adata = zeroarray(m, p);
        udata = zeroarray(m, p);
        ygap_ind = indexin([:ygap], m.variables)[1]
        adata[sim[1:s], ygap_ind] .= -0.10
        udata[sim[1:s], ygap_ind] .= -0.10
        adata[term,:] .= atrue[term,:]
        udata[term,:] .= utrue[term,:]
        exogenize!(p, :ygap, sim[1:s])
        endogenize!(p, :ygap_shk, sim[1:s])
        # Run the simulations and test
        ares = simulate(m, p, adata)
        @test ares ≈ atrue atol = m.options.tol
        ures = simulate(m, p, udata; anticipate=false)
        @test ures ≈ utrue atol = m.options.tol
    end
end

@testset "FCTypes" begin
    let m = Model()
        @variables m x
        @equations m begin x[t] = x[t + 3] end
        @initialize m
        p = Plan(m, 1:1)
        ed = zerodata(m, p)
        @test_throws ArgumentError simulate(m, p, ed, fctype=fcnatural)
    end
    let m = E1.model
        p = Plan(m, 3:17)
        ed = zerodata(m, p)
        ed .= rand(Float64, size(ed))
        @test_logs (:error, r"The system is underdetermined.*") (@test_throws SingularException simulate(m, p, ed, fctype=fcnatural))
        sd = StateSpaceEcon.StackedTimeSolver.StackedTimeSolverData(m, p, fcgiven)
        @test_throws ErrorException StateSpaceEcon.StackedTimeSolver.update_plan!(sd, m, Plan(m, 3:8))
        x = rand(Float64, size(ed))
        R1, _ = StateSpaceEcon.StackedTimeSolver.global_RJ(x, ed, sd)
        R2 = StateSpaceEcon.StackedTimeSolver.global_R!(similar(R1), x, ed, sd)
        @test R1 ≈ R2
    end
    let m = E3.model
        p = Plan(m, 3:177)
        ed = zerodata(m, p)
        ed[3U, m.shocks] .= 0.2 * rand(1,3)
        ed[3U:end, m.variables] .= rand(178, 3)
        empty!(m.sstate.constraints)
        clear_sstate!(m)
        @test_throws ArgumentError simulate(m, p, ed, fctype=fclevel)
        @test_throws ArgumentError simulate(m, p, ed, fctype=fcslope)
        vec1 = sssolve!(m)
        initial_sstate!(m, 0.4 * rand(Float64, size(m.sstate.values)))
        vec2 = sssolve!(m)
        @test vec1 ≈ vec2
        s1 = simulate(m, p, ed, fctype=fclevel)
        s2 = simulate(m, p, ed, fctype=fcslope)
        s3 = simulate(m, p, ed, fctype=fcnatural)
        @test maximum(abs, s1 - s2; dims = :) < 1e-8
        @test maximum(abs, s1 - s3; dims = :) < 1e-8
        @test maximum(abs, s3 - s2; dims = :) < 1e-8

        sd = StateSpaceEcon.StackedTimeSolver.StackedTimeSolverData(m, p, fcgiven)
        x = rand(Float64, size(ed))
        R1, _ = StateSpaceEcon.StackedTimeSolver.global_RJ(x, ed, sd)
        R2 = StateSpaceEcon.StackedTimeSolver.global_R!(similar(R1), x, ed, sd)
        @test R1 ≈ R2
    end
    let m = E6.model
        empty!(m.sstate.constraints)
        @steadystate m lp  = 1.0
        @steadystate m lp + lyn = 1.5
        @steadystate m lp + ly = 1.2
        clear_sstate!(m)
        sssolve!(m)
        p = Plan(m, 3:177)
        ed = zerodata(m, p)
        ed.dlp_shk[3] = rand()
        ed.dly_shk[3] = rand()
        s2 = simulate(m, p, ed, fctype=fcslope)
        s3 = simulate(m, p, ed, fctype=fcnatural)
        @test maximum(abs, s3 - s2; dims = :) < 1e-12

        sd = StateSpaceEcon.StackedTimeSolver.StackedTimeSolverData(m, p, fcgiven)
        x = rand(Float64, size(ed))
        R1, _ = StateSpaceEcon.StackedTimeSolver.global_RJ(x, ed, sd)
        R2 = StateSpaceEcon.StackedTimeSolver.global_R!(similar(R1), x, ed, sd)
    end
end