
using LinearAlgebra
using JLD2
@testset "shkdcmp" begin
    m = deepcopy(E7A.model)
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
    result = shockdecomp(m, p, exog_data; fctype=fcnatural)
    @test compare(result, expected; atol=1.0e-9, quiet=true)

    # shock on dly is half inventory half consumption
    dlinv_shk = result.sd.dlinv.dlinv_shk[result.sd.dlinv.dlinv_shk.>1e-9] # positive shock values
    dly_dlinv_shk = result.sd.dly.dlinv_shk[result.sd.dlinv.dlinv_shk.>1e-9] # same rows for dly
    @test all(dly_dlinv_shk ./ dlinv_shk .≈ 0.5)
    dlc_shk = result.sd.dlc.dlc_shk[result.sd.dlc.dlc_shk.>1e-9] # positive shock values
    dly_dlc_shk = result.sd.dly.dlc_shk[result.sd.dlc.dlc_shk.>1e-9] # same rows for dly
    @test all(dly_dlc_shk ./ dlc_shk .≈ 0.5)

    sd_zeros = fill!(similar(result.sd.dlc), 0)

    # for consumption init, term, and non-linear are all zero
    @test isapprox(result.sd.dlc.init, sd_zeros[:, 1], atol=1e-12)
    @test isapprox(result.sd.dlc.term, sd_zeros[:, 1], atol=1e-12)
    @test isapprox(result.sd.dlc.nonlinear, sd_zeros[:, 1], atol=1e-12)

    # for output init and term are zero, but non-linear is not
    @test isapprox(result.sd.dly.init, sd_zeros[:, 1], atol=1e-12)
    @test isapprox(result.sd.dly.term, sd_zeros[:, 1], atol=1e-12)
    @test isapprox(result.sd.dly.nonlinear, sd_zeros[:, 1], atol=1e-12) == false


    # deviation shocks should be identical
    result_dev = shockdecomp(m, p, exog_data .- steadystatedata(m, p); fctype=fcnatural, deviation=true)
    @test compare(result_dev.c, expected.c; atol=1.0e-9, quiet=true)
    @test compare(result_dev.sd, expected.sd; atol=1.0e-9, quiet=true)
    @test compare(result_dev.s .+ steadystatedata(m, p), expected.s; atol=1.0e-9, quiet=true)

    # but we still have the same main shocks:
    @test isapprox(result_dev.sd.dly.dlc_shk - result.sd.dly.dlc_shk, sd_zeros[:, 1], atol=1e-12)
    @test isapprox(result_dev.sd.dly.dlinv_shk - result.sd.dly.dlinv_shk, sd_zeros[:, 1], atol=1e-12)
    @test isapprox(result_dev.sd.dlc.dlc_shk - result.sd.dlc.dlc_shk, sd_zeros[:, 1], atol=1e-12)
    @test isapprox(result_dev.sd.dlinv.dlinv_shk - result.sd.dlinv.dlinv_shk, sd_zeros[:, 1], atol=1e-12)

    # and the same relation between shocks
    dlinv_shk = result_dev.sd.dlinv.dlinv_shk[result_dev.sd.dlinv.dlinv_shk.>1e-9] # positive shock values
    dly_dlinv_shk = result_dev.sd.dly.dlinv_shk[result_dev.sd.dlinv.dlinv_shk.>1e-9] # same rows for dly
    @test all(dly_dlinv_shk ./ dlinv_shk .≈ 0.5)
    dlc_shk = result_dev.sd.dlc.dlc_shk[result_dev.sd.dlc.dlc_shk.>1e-9] # positive shock values
    dly_dlc_shk = result_dev.sd.dly.dlc_shk[result_dev.sd.dlc.dlc_shk.>1e-9] # same rows for dly
    @test all(dly_dlc_shk ./ dlc_shk .≈ 0.5)

    # for consumption, term, init and nonlinear are zero
    @test norm(result_dev.sd.dlc.init, Inf) < 1e-12
    @test norm(result_dev.sd.dlc.term, Inf) < 1e-12
    @test norm(result_dev.sd.dlc.nonlinear, Inf) < 1e-12

    # output is the same
    @test norm(result_dev.sd.dly.init, Inf) < 1e-12
    @test norm(result_dev.sd.dly.term, Inf) < 1e-12
    # some non-linearity accumulation in dly
    # @test norm(result_dev.sd.dly.nonlinear, Inf) < 1e-12


    #=  disable these tests - not true with this non-linear model.  Maybe will test with linearized version 
        # providing the shocked data as control and applying the same shock produced the same decomp
        exog_data2 = copy(exog_data)
        exog_data2[2021Q1, m.shocks] .+= 0.1
        result2 = nothing
        out = @capture_err begin
            result2 = shockdecomp(m, p, exog_data2; control=result.s, fctype = fcnatural)
        end
        for var in m.variables
            @test compare(result2.sd[var][m.shocks], expected.sd[var][m.shocks]; atol=1.0e-9, quiet=true)
        end
        # but the comparison produces a warning:
        @test occursin("Control is not a solution", out)


        # in deviation space (where the control is subtracted), the shocks are twice as large
        result2_dev = nothing
        out = @capture_err begin
            result2_dev = shockdecomp(m, p, exog_data2; control=exog_data, deviation=true, fctype = fcnatural)
        end
        for var in m.variables
            @test compare(result2_dev.sd[var][m.shocks], expected.sd[var][m.shocks] .* 2; atol=1.0e-9, quiet=true)
        end
        @test occursin("Control is not a solution", out)
     =#

end

