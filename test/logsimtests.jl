
@testset "logeqn" begin
    let m = Model()
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
        @test m.sstate.X.level ≈ 1.0 
        @test m.sstate.X.slope ≈ m.rate
        @test m.sstate.Y.level + 1 ≈ 0.0 + 1 
        @test m.sstate.Y.slope + 1 ≈ m.rate + 1 

        p = Plan(m, 1U:10U)

        # make sure the steadystate data is built correctly
        a = steadystatearray(m, p)
        d = steadystatedict(m, p)
        k = steadystatedata(m, p)

        @test a == k
        for s in Symbol.(m.allvars)
            @test k.:($s) == d[string(s)]
        end
        @test k.X[1U] ≈ 1.0
        @test k.Y[1U] + 1 ≈ 1.0 
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

        for fc in (fcgiven, fclevel, fcslope, fcnatural)
            @test simulate(m, ed, p, fctype=fc) ≈ k
        end

    end

    let m = Model()
        m.tol = 1e-15
        @variables m @log(X), Y
        @shocks m Ex Ey
        @parameters m rate = 1.015
        @equations m begin
            @log X[t + 1] = rate * X[t] + Ex[t]
            Y[t + 1] = Y[t] + rate + Ey[t]
        end

        @initialize m
        @steadystate m X = 1.0
        @steadystate m Y = 0.0

        clear_sstate!(m)
        sssolve!(m)
        @test m.sstate.X.level + 1 ≈ 1.0 + 1
        @test m.sstate.X.slope + 1 ≈ m.rate + 1
        @test m.sstate.Y.level + 1 ≈ 0.0 + 1 
        @test m.sstate.Y.slope + 1 ≈ m.rate + 1 

        p = Plan(m, 1U:10U)

        # make sure the steadystate data is built correctly
        a = steadystatearray(m, p)
        d = steadystatedict(m, p)
        k = steadystatedata(m, p)

        @test a == k
        for s in Symbol.(m.allvars)
            @test k.:($s) == d[string(s)]
        end
        @test k.X[1U] + 1 ≈ 1.0 + 1
        @test k.Y[1U] + 1 ≈ 0.0 + 1
        @test pct(k.X, -1) ≈ TSeries(p.range, (m.rate - 1) * 100)
        @test diff(k.Y) ≈ TSeries(p.range, m.rate)

        # make sure simulations do as expected

        # this model has no final conditions (maxlag == 0)
        ed = SimData(firstdate(p), m.allvars, rand(Float64, (length(p), length(m.allvars))))
        # no shocks
        ed.Ex .= 0.0
        ed.Ey .= 0.0
        # Initial conditions don't matter. We set final conditions
        ed[lastdate(p)] = k[lastdate(p)]

        @test simulate(m, ed, p, fctype=fcgiven) ≈ k
        @test simulate(m, ed, p, fctype=fclevel) ≈ k

        # system is inconsistent for fcslope and fcnatural
        # we do some exo-endo juggling to get unique solution

        # we need exogenous variables somewhere
        exogenize!(p, m.variables, firstdate(p))
        # we need endogenous shocks on the last simulation period
        endogenize!(p, m.shocks, lastdate(p) - 1)

        # update exogenous data accordingly
        ed[firstdate(p)] = k[firstdate(p)]
        ed[lastdate(p),1:2] .= rand(2)

        @test simulate(m, ed, p, fctype=fcslope) ≈ k
        @test simulate(m, ed, p, fctype=fcnatural) ≈ k

    end

end
