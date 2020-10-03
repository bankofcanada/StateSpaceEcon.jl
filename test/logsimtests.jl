
@testset "logeqn" begin
    m = Model()
    m.tol = 1e-15
    @variables m @log(X), Y
    @shocks m Ex Ey
    @parameters m rate = 1.015
    @equations m begin
        @log X[t] = rate * X[t - 1] + Ex[t]
        Y[t] = Y[t - 1] + rate + Ey[t]
    end

    @initialize m
    @steadystate m X = 1.0
    @steadystate m Y = 0.0

    clear_sstate!(m)
    sssolve!(m)
    @test m.sstate.X.level + 1 ≈ log(1.0) + 1 
    @test m.sstate.X.slope + 1 ≈ log(m.rate) + 1 
    @test m.sstate.Y.level + 1 ≈ 0.0 + 1 
    @test m.sstate.Y.slope + 1 ≈ m.rate + 1 

    p = Plan(m, 1:10)

    # make sure the steadystate data is built correctly
    a = steadystatearray(m, p)
    d = steadystatedict(m, p)
    k = steadystatedata(m, p)

    @test a == k
    for s in Symbol.(m.allvars)
        @test k.:($s) == d[string(s)]
    end
    @test k.X[1] ≈ 1.0
    @test k.Y[1] + 1 ≈ 1.0 
    @test pct(k.X, -1) ≈ TSeries(p.range, (m.rate - 1) * 100)
    @test diff(k.Y) ≈ TSeries(p.range, m.rate)

    # make sure simulations do as expected

    # this model has no final conditions (maxlag == 0)
    ed = SimData(firstdate(p), m.allvars, rand(Float64, (length(p), length(m.allvars))))
    # no shocks
    ed.Ex .= 0.0
    ed.Ey .= 0.0
    # initial 
    ed[firstdate(p)] = k[firstdate(p)]

    @test simulate(m, ed, p, fctype=fcgiven) ≈ k
    @test simulate(m, ed, p, fctype=fclevel) ≈ k

    exogenize!(p, m.variables, firstdate(p))
    endogenize!(p, m.shocks, lastdate(p) - 1)
    ed[firstdate(p)] = k[firstdate(p)]
    ed[lastdate(p),1:2] .= rand(2)

    @test simulate(m, ed, p, fctype=fcslope) ≈ k
    @test simulate(m, ed, p, fctype=fcnatural) ≈ k

end
