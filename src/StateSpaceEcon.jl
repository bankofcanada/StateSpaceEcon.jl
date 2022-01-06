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

include("simdata.jl")
include("misc.jl")
include("SteadyStateSolver.jl")
include("Plans.jl")
include("plandata.jl")
include("StackedTimeSolver.jl")

end # module
