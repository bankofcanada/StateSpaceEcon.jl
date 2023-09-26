##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
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

using TimerOutputs
const timer = TimerOutput()

using ..Plans

import ..steadystatearray
import ..SimData
import ..rawdata

import ..SimFailed
import ..isfailed
import ..MaybeSimData

import ModelBaseEcon.hasevaldata
import ModelBaseEcon.getevaldata
import ModelBaseEcon.setevaldata!
import ModelBaseEcon.hassolverdata
import ModelBaseEcon.getsolverdata
import ModelBaseEcon.setsolverdata!

using Pardiso


using SuiteSparse
using SuiteSparse.UMFPACK

include("stackedtime/sparse.jl")

include("stackedtime/fctypes.jl")
include("stackedtime/misc.jl")
include("stackedtime/solverdata.jl")
include("stackedtime/sim_lm.jl")
include("stackedtime/sim_nr.jl")
include("stackedtime/sim_gn.jl")
include("stackedtime/simulate.jl")
include("stackedtime/shockdecomp.jl")
include("stackedtime/stoch_simulate.jl")

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

export use_pardiso, use_pardiso!
export use_umfpack, use_umfpack!

# the following are deprecated
export seriesoverlay, dictoverlay

