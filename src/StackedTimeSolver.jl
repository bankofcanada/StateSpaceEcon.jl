##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

"""
    StackedTimeSolver

A module that is part of StateSpaceEcon package.
Contains methods for solving the dynamic system of equations
for the model and running simulations.
"""
module StackedTimeSolver

using SparseArrays
using LinearAlgebra
using ForwardDiff
using DiffResults
using Printf

using ModelBaseEcon
using TimeSeriesEcon

using ..Plans

include("stackedtime/abstract.jl")
include("stackedtime/fctypes.jl")
include("stackedtime/misc.jl")
include("stackedtime/solverdata.jl")
include("stackedtime/simulate.jl")

end # module

using .StackedTimeSolver

export FinalCondition
export NoFinalCondition
export HasFinalCondition
export FCNone, fcnone
export FCGiven, fcgiven
export FCMatchSSLevel, fclevel
export FCMatchSSRate, fcslope, fcrate
export FCConstRate, fcnatural
export setfc

export simulate
export seriesoverlay, dictoverlay
export dict2array, array2dict, array2data
