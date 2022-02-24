
using JLD2
@testset "shkdcmp" begin
    m = E7A.model
    expected = Workspace(load("data/shkdcmp_E7A.jld2"))

    empty!(m.sstate.constraints)
    @steadystate m lc = 1
    @steadystate m linv = 1
    clear_sstate!(m)
    sssolve!(m)
    @test 0 == check_sstate(m)

    p = Plan(m, 2021Q1:2035Q4)
    exog_data = steadystatedata(m, p)
    exog_data[2021Q1, m.shocks] .+= 0.1
    result = shockdecomp(m, p, exog_data; fctype = fcnatural)
    @test @compare result expected quiet
end

