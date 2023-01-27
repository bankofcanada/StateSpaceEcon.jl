
using JLD2
@testset "shkdcmp" begin
    m = deepcopy(E7A.model)
    expected = Workspace(load("data/shkdcmp_E7A.jld2"))
    TimeSeriesEcon.clean_old_frequencies!(expected)

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
    @test compare(result, expected; atol=1.0e-9, quiet=true)

    # shock on dly is half inventory half consumption
    dlinv_shk = result.sd.dlinv.dlinv_shk[result.sd.dlinv.dlinv_shk .> 1e-9]; # positive shock values
    dly_dlinv_shk = result.sd.dly.dlinv_shk[result.sd.dlinv.dlinv_shk .> 1e-9]; # same rows for dly
    @test all(dly_dlinv_shk ./ dlinv_shk .≈ 0.5) == 1
    dlc_shk = result.sd.dlc.dlc_shk[result.sd.dlc.dlc_shk .> 1e-9]; # positive shock values
    dly_dlc_shk = result.sd.dly.dlc_shk[result.sd.dlc.dlc_shk .> 1e-9]; # same rows for dly
    @test all(dly_dlc_shk ./ dlc_shk .≈ 0.5) == 1

    sd_zeros = similar(result.sd.dlc)
    sd_zeros .= 0

    # for consumption init, term, and non-linear are all zero
    @test isapprox(result.sd.dlc.init, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result.sd.dlc.term, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result.sd.dlc.nonlinear, sd_zeros[:,1], atol=1e-12)

    # for output init and term are zero, but non-linear is not
    @test isapprox(result.sd.dly.init, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result.sd.dly.term, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result.sd.dly.nonlinear, sd_zeros[:,1], atol=1e-12) == false


    # deviation shocks are not identical
    result_dev = shockdecomp(m, p, exog_data; fctype = fcnatural, deviation=true);
    @suppress begin
        @test compare(result_dev, expected; atol=1.0e-9, quiet=true) == false;
    end

    # but we still have the same main shocks:
    @test isapprox(result_dev.sd.dly.dlc_shk - result.sd.dly.dlc_shk, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result_dev.sd.dly.dlinv_shk - result.sd.dly.dlinv_shk, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result_dev.sd.dlc.dlc_shk - result.sd.dlc.dlc_shk, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result_dev.sd.dlinv.dlinv_shk - result.sd.dlinv.dlinv_shk, sd_zeros[:,1], atol=1e-12)

    # and the same relation between shocks
    dlinv_shk = result_dev.sd.dlinv.dlinv_shk[result_dev.sd.dlinv.dlinv_shk .> 1e-9]; # positive shock values
    dly_dlinv_shk = result_dev.sd.dly.dlinv_shk[result_dev.sd.dlinv.dlinv_shk .> 1e-9]; # same rows for dly
    @test all(dly_dlinv_shk ./ dlinv_shk .≈ 0.5) == 1;
    dlc_shk = result_dev.sd.dlc.dlc_shk[result_dev.sd.dlc.dlc_shk .> 1e-9]; # positive shock values
    dly_dlc_shk = result_dev.sd.dly.dlc_shk[result_dev.sd.dlc.dlc_shk .> 1e-9]; # same rows for dly
    @test all(dly_dlc_shk ./ dlc_shk .≈ 0.5) == 1;

    # for consumption, term is zero, but init and nonlinear are not
    @test isapprox(result_dev.sd.dlc.init, sd_zeros[:,1], atol=1e-12) == false
    @test isapprox(result_dev.sd.dlc.term, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result_dev.sd.dlc.nonlinear, sd_zeros[:,1], atol=1e-12) == false

    # output is the same
    @test isapprox(result_dev.sd.dly.init, sd_zeros[:,1], atol=1e-12) == false
    @test isapprox(result_dev.sd.dly.term, sd_zeros[:,1], atol=1e-12)
    @test isapprox(result_dev.sd.dly.nonlinear, sd_zeros[:,1], atol=1e-12) == false


    # providing the shocked data as control and applying the same shock produced the same decomp
    exog_data2 = deepcopy(exog_data)
    exog_data2[2021Q1, m.shocks] .+= 0.1
    result2 = nothing
    out = @capture_err begin
        result2 = shockdecomp(m, p, exog_data2; control=exog_data, fctype = fcnatural)
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

end

