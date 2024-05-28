##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

"""
    StateSpaceEcon

A package for Macroeconomic modelling.

"""
module StateSpaceEcon

using TimeSeriesEcon
using ModelBaseEcon
using ModelBaseEcon.OrderedCollections

include("Kalman.jl")
include("misc.jl")
include("SteadyStateSolver.jl")
include("Plans.jl")
include("simdata.jl")
include("plandata.jl")
include("StackedTimeSolver.jl")
include("FirstOrderSolver.jl")

const _solvers = LittleDict{Symbol, Module}(
    :stackedtime => StackedTimeSolver,
    :firstorder => FirstOrderSolver
)

function getsolvermodule(solvername::Symbol) 
    SolverModule = get(_solvers, solvername, nothing)
    if SolverModule === nothing
        error(ArgumentError("Unknown solver :$solvername."))
    end
    return SolverModule
end

include("simulate.jl")

include("DFMSolver.jl")

end # module
