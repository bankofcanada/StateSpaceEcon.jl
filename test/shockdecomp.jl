
using LinearAlgebra
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

    rng = 2021Q1:2035Q4
    p = Plan(m, rng)
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


    m.p_dlinv_ss = 0
    m.p_dlc_ss = 0
    m.delta = 0.6
    clear_sstate!(m)
    sssolve!(m)
    @test 0 == check_sstate(m)

    linearize!(m)
    solve!(m, solver=:firstorder)

    exog_data3 = steadystatedata(m, p)
    exog_data3[first(rng), m.shocks] .+= 0.1
    result3 = shockdecomp(m, p, exog_data3; variant=:linearize, fctype=fcnatural)

    foreach(result3.sd) do ((var, sd))
        @test norm(sd.nonlinear, Inf) < 1e-12
        @test norm(sd.init, Inf) < 1e-12
        @test norm(sd.term, Inf) < 1e-12
    end
    @test norm(result3.sd.dlc.dlinv_shk, Inf) < 1e-12
    @test norm(result3.sd.dlinv.dlc_shk, Inf) < 1e-12

    @test all(result3.sd) do ((var, sd))
        norm(sd.init, Inf) < 1e-12
    end
    @test all(result3.sd) do ((var, sd))
        norm(sd.term, Inf) < 1e-12
    end

    # for linearized model with stacked time (result3) must be the same as firstorder (result4)
    result3fo = shockdecomp(m, p, exog_data3; solver=:firstorder)
    @test compare(result3, result3fo; ignoremissing=true, atol=2^10 * eps(), quiet=true)

    # additive for linearized model 
    exog_data4 = copy(exog_data3)
    exog_data4[first(rng), m.shocks] .+= 0.1
    result4 = shockdecomp(m, p, exog_data4; control=result3.s, variant=:linearize, fctype=fcnatural)

    @test result3.s ≈ result4.c
    @test result3.s - result3.c ≈ result4.s - result4.c
    @test compare(result3.sd, result4.sd, ignoremissing=true, atol=2^10 * eps(), quiet=true)

    result4fo = shockdecomp(m, p, exog_data4; control=result3.s, solver=:firstorder)
    @test compare(result4, result4fo; ignoremissing=true, atol=2^10 * eps(), quiet=true)

end

