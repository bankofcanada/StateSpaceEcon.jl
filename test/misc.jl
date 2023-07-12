##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

@testset "printmatrix" begin
    io = IOBuffer()
    printmatrix(io, rand(3, 4))
    seekstart(io)
    @test length(readlines(io)) == 3
end

@testset "fcset" begin
    m = getE7A()

    empty!(m.sstate.constraints)
    @steadystate m lc = 1
    @steadystate m linv = 1
    clear_sstate!(m)
    sssolve!(m)
    @test 0 == check_sstate(m)

    p = Plan(m, 2021Q1:2035Q4)
    exog_data = steadystatedata(m, p)
    exog_data[2021Q1, m.shocks] .+= 0.1

    gdata = StateSpaceEcon.StackedTimeSolver.StackedTimeSolverData(m, p, setfc(m, fcnatural))

    @test length(gdata.FC) == 9
    @test length(filter(x -> x == fcnatural, gdata.FC)) == 7
    @test length(filter(x -> x == fcnone, gdata.FC)) == 2

    # change the FC
    StateSpaceEcon.setfc!(gdata.FC, m, :dlc, fclevel)

    @test length(gdata.FC) == 9
    @test length(filter(x -> x == fcnatural, gdata.FC)) == 6
    @test length(filter(x -> x == fclevel, gdata.FC)) == 1
    @test length(filter(x -> x == fcnone, gdata.FC)) == 2
end
