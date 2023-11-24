##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

rng = Random.default_rng()
@testset "sim_lm" begin
    m = getE7A()
    @test begin
        empty!(m.sstate.constraints)
        @steadystate m linv = lc - 7
        @steadystate m lc = 14
        clear_sstate!(m)
        sssolve!(m)
        issssolved(m) && (0 == check_sstate(m))
    end
    p01 = Plan(m, 1:100)
    data01 = steadystatedata(m, p01)
    data01[begin:0U, m.variables] .+= randn(rng, m.maxlag, m.nvars)
    data01[1U:10U, m.shocks] .+= randn(rng, 10, m.nshks)

    res01 = []
    for sim_solver in (:sim_nr, :sim_lm, :sim_gn)
        tmp = simulate(m, p01, data01; warn_maxiter=:warn, sim_solver, verbose=false, fctype=fcnatural)
        push!(res01, tmp)
    end
    
    @test @compare res01[1] res01[2] quiet
    @test @compare res01[1] res01[3] quiet
    @test @compare res01[2] res01[3] quiet

end


