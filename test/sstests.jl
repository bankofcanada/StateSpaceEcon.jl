
@testset "E1.sstate" begin
    let m = E1.model
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == falses(2)
        # 
        @steadystate m y = 1.2
        m.α = 0.5
        m.β = 1.0 - m.α
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == [true, false]
        @test m.sstate.values[1] == 1.2
        # 
        m.α = 0.4
        m.β = 1.0 - m.α
        clear_sstate!(m, verbose = false)
        @test m.sstate.mask == trues(2)
        sssolve!(m; verbose = false)
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(2)
        @test m.sstate.values == [1.2, 0.0]
        # 
        empty!(m.sstate.constraints)
        m.α = 0.3
        m.β = 1.0 - m.α
        clear_sstate!(m, verbose = false)
        sssolve!(m; verbose = false)
        @test check_sstate(m) == 0
        @test m.sstate.values[2] == 0.0
    end
end

@testset "E2.sstate" begin
    let m = E2.model
        clear_sstate!(m)
        sssolve!(m);
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(6)
        @test m.sstate.values == zeros(6)

        empty!(m.sstate.constraints)
        @steadystate m rate = 0.0
        clear_sstate!(m)
        @test m.sstate.mask[3] == true
        sssolve!(m)
        @test check_sstate(m) == 0
        @test m.sstate.mask == trues(6)
        @test m.sstate.values == zeros(6)

        empty!(m.sstate.constraints)
        @steadystate m rate = 0.1
        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) > 0
    end
end

@testset "E6.sstate" begin
    let m = E6.model
        m.options.maxiter = 50

        clear_sstate!(m)
        sssolve!(m)
        @test check_sstate(m) == 0
        @test sum(m.sstate.mask) == 9
        @test m.sstate.values[m.sstate.mask] ≈ [0.005, 0.0, 0.0045, 0.0, 0.0095, 0.0, 0.005, 0.0045, 0.0095]

        empty!(m.sstate.constraints)
        @steadystate m lp = 2
        @steadystate m ly = 3
        @steadystate m lyn = 7
        clear_sstate!(m)
        sssolve!(m);
        @test check_sstate(m) == 0
        @test all(m.sstate.mask)
        @test m.sstate.values ≈ [0.005, 0.0, 0.0045, 0.0, 0.0095, 0.0, 2.0, 0.005, 3.0, 0.0045, 7.0, 0.0095]
    end
end

@testset "E7.sstate" begin
    let m = E7.model
        m.options.maxiter = 100
        m.options.tol = 1e-9

        clear_sstate!(m)
        sssolve!(m);
        @test check_sstate(m) == 0
        # underdetermined system - two free variables
        @test sum(m.sstate.mask) == length(m.sstate.mask) - 2

        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7;
        @steadystate m lc = 14;
        clear_sstate!(m)
        sssolve!(m);
        @test check_sstate(m) == 0
        @test all(m.sstate.mask)
        @test m.sstate.values ≈ [0.004, 0.0, 0.004, 0.0, 0.004, 0.0, 14.0, 0.004, 7.0, 0.004, 9.267287357063445, 0.004, 14.000911466453774, 0.004, 14.000911466453774, 0.004, 9.267287357063445, 0.004]

        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        @steadystate m linv = 8
        clear_sstate!(m)
        sssolve!(m);
        # overdetermined inconsistent set of constraints
        @test check_sstate(m) > 0

        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        @steadystate m ly = log(exp(lc) + exp(linv));
        clear_sstate!(m)
        sssolve!(m);
        # overdetermined consistent set of constraints
        @test check_sstate(m) == 0
        @test all(m.sstate.mask)
        @test m.sstate.values ≈ [0.004, 0.0, 0.004, 0.0, 0.004, 0.0, 14.0, 0.004, 7.0, 0.004, 9.267287357063445, 0.004, 14.000911466453774, 0.004, 14.000911466453774, 0.004, 9.267287357063445, 0.004]
    end
end