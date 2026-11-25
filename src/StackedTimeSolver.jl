##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

"""
    StackedTimeSolver

A module that is part of StateSpaceEcon package.
Contains methods for solving the dynamic system of equations
for the model and running simulations.
"""
module StackedTimeSolver

using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, nzrange, rowvals, sparse
using LinearAlgebra: LinearAlgebra, Factorization, LAPACKException, SingularException, ldiv!, lu, lu!, norm
using ForwardDiff: ForwardDiff
using DiffResults: DiffResults
using Printf: Printf

using ModelBaseEcon: ModelBaseEcon, Model, ModelVariable, Options, eval_R!, eval_RJ, getoption, inverse_transform,
    isexog, islin, islog, isneglog, isshock, issssolved, refresh_med!, shocks, transform
using TimeSeriesEcon: TimeSeriesEcon, MVTSeries, TSeries, U, Workspace, colnames, firstdate, lastdate, overlay, rangeof, rawdata

using TimerOutputs: TimerOutputs, @timeit_debug, TimerOutput
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

using Pardiso: Pardiso, MKLPardisoSolver, fix_iparm!, get_iparm, get_matrix, pardiso, pardisoinit, set_iparm!,
    set_matrixtype!, set_phase!


using SparseArrays.UMFPACK: UMFPACK

include("stackedtime/sparse.jl")

include("stackedtime/fctypes.jl")
include("stackedtime/misc.jl")
include("stackedtime/solverdata.jl")
include("stackedtime/sim_nr.jl")
include("stackedtime/sim_lm.jl")
include("stackedtime/sim_gn.jl")
include("stackedtime/simulate.jl")
include("stackedtime/shockdecomp.jl")
include("stackedtime/stoch_simulate.jl")

end # module

using .StackedTimeSolver

export FinalCondition
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

