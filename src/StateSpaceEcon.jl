##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

"""
    StateSpaceEcon

A package for Macroeconomic modelling.

"""
module StateSpaceEcon

using TimeSeriesEcon
using ModelBaseEcon
# using ModelBaseEcon.OptionsMod
# using ModelBaseEcon.Timer

include("simdata.jl")
include("misc.jl")
include("SteadyStateSolver.jl")
include("Plans.jl")
include("plandata.jl")
include("StackedTimeSolver.jl")
include("plotrecipes.jl")

end # module
