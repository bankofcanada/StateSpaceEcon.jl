##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    StateSpaceEcon

A package for Macroeconomic modelling.

"""
module StateSpaceEcon

using TimeSeriesEcon
using ModelBaseEcon

include("misc.jl")
include("SteadyStateSolver.jl")
include("Plans.jl")
include("simdata.jl")
include("plandata.jl")
include("StackedTimeSolver.jl")
include("FirstOrderSolver.jl")
include("solve.jl")
include("simulate.jl")

end # module
