using ModelingToolkit
using NonlinearSolve
using SymbolicIndexingInterface
using DelimitedFiles
using TimeSeriesEcon

function test_mtk(m_in, path; atol=1e-9)

    plan_data = readdlm(path, ',', Float64)

    for fctype in [fcgiven, fclevel, fcslope, fcnatural]

        m1 = deepcopy(m_in)
        m2 = deepcopy(m_in)

        if fctype === fclevel || fctype === fcslope

            # Compute steady state with MTK.
            sss = steady_state_system(m1)
            solve_steady_state!(m1, sss; solver = NewtonRaphson())

            # Compute steady state with StateSpaceEcon.
            clear_sstate!(m2)
            sssolve!(m2)

            @test isapprox(m1.sstate.values, m2.sstate.values; atol)

        end

        # Simulate with MTK.
        IDX = size(plan_data, 1)
        plan = Plan(m1, 1+m1.maxlag:IDX-m1.maxlead)
        s = stacked_time_system(m1, plan_data; fctype, plan)
        nf = NonlinearFunction(s)
        prob = NonlinearProblem(nf, zeros(length(unknowns(s))))
        sol = solve(prob, NewtonRaphson())
        result = getu(sol, reduce(vcat, map(collect, map(var -> getproperty(s, var.name), m1.variables))))(sol)
        result = reshape(result, size(plan_data, 1) - m1.maxlag - m1.maxlead, :)

        # Simulate with StateSpaceEcon.
        nvars = nvariables(m2)
        nshks = nshocks(m2)
        IDX = size(plan_data, 1)
        plan = Plan(m2, 1+m2.maxlag:IDX-m2.maxlead)
        data = zeroarray(m2, plan)
        data[m2.maxlag+1:end-m2.maxlead, nvars.+(1:nshks)] = plan_data[m2.maxlag+1:end-m2.maxlead, nvars.+(1:nshks)] # Shocks
        data[1:m2.maxlag, :] = plan_data[1:m2.maxlag, :] # Initial conditions
        if fctype === fcgiven
            data[end-m2.maxlead+1:end, :] = plan_data[end-m2.maxlead+1:end, :] # Final conditions
        end
        correct = simulate(m2, plan, data; fctype)
        correct_mvts = MVTSeries(1U:(IDX)U, m2.varshks, correct)

        result_mvts = sol2data(sol, s, m1, plan)
        
        @test isapprox(result, correct[m2.maxlag+1:end-m2.maxlead, 1:nvars]; atol)
        @test isapprox(result_mvts[begin:end-m_in.maxlag-m_in.maxlead-1, :], correct_mvts[begin:end-m_in.maxlag-m_in.maxlead-1, :]; atol)        

        # test with adjusted plan
        # Simulate with MTK.
        IDX = size(plan_data, 1)
        plan = Plan(m1, 1+m1.maxlag:IDX-m1.maxlead)
        autoexogenize!(plan, m1, first(rangeof(plan)):50U)
        s = stacked_time_system(m1, plan_data; fctype, plan)
        nf = NonlinearFunction(s)
        prob = NonlinearProblem(nf, zeros(length(unknowns(s))))
        sol = solve(prob, NewtonRaphson())
        result = getu(sol, reduce(vcat, map(collect, map(var -> getproperty(s, var.name), m1.variables))))(sol)
        result_mvts2 = sol2data(sol, s, m1, plan)
        # result = reshape(result, size(plan_data, 1) - m1.maxlag - m1.maxlead, :)

        # Simulate with StateSpaceEcon.
        nvars = nvariables(m2)
        nshks = nshocks(m2)
        IDX = size(plan_data, 1)
        plan = Plan(m2, 1+m2.maxlag:IDX-m2.maxlead)
        autoexogenize!(plan, m2, 1U:50U)
        data = zeroarray(m2, plan)
        data[m2.maxlag+1:50, 1:nvars] = plan_data[m2.maxlag+1:50, 1:nvars] # Variables (exogenous part)
        data[51:end-m2.maxlead, nvars.+(1:nshks)] = plan_data[51:end-m2.maxlead, nvars.+(1:nshks)] # Shocks (exogenous part)
        data[1:m2.maxlag, :] = plan_data[1:m2.maxlag, :] # Initial conditions
        if fctype === fcgiven
            data[end-m2.maxlead+1:end, :] = plan_data[end-m2.maxlead+1:end, :] # Final conditions
        end
        correct = simulate(m2, plan, data; fctype)
        correct_mvts2 = MVTSeries(1U:(IDX)U, m2.varshks, correct)

        @test isapprox(result_mvts2[begin:end-m_in.maxlag-m_in.maxlead-1, :], correct_mvts2[begin:end-m_in.maxlag-m_in.maxlead-1, :]; atol)
    end

end

@testset "E3.mtkext" begin
    test_mtk(getE3(), joinpath(@__DIR__, "data", "M3_TestData.csv"))
end

@testset "E7A.mtkext" begin
    test_mtk(getE7A(), joinpath(@__DIR__, "data", "M7_TestData.csv"))
end
