##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

module FirstOrderSolver

using SparseArrays: SparseArrays, SparseMatrixCSC, findnz
using LinearAlgebra: LinearAlgebra, BLAS, Factorization, LAPACK, ldiv!, lu, rdiv!
using ForwardDiff: ForwardDiff
using DiffResults: DiffResults
using Printf: Printf

using ModelBaseEcon: ModelBaseEcon, Model, inverse_transform, isexog, islinearized, islog, isneglog, isshock,
    linearize!, refresh_med!, transform
using TimeSeriesEcon: TimeSeriesEcon, MVTSeries, Workspace, lag, overlay

import ModelBaseEcon.getevaldata
import ModelBaseEcon.setevaldata!
import ModelBaseEcon.LinearizedModelEvaluationData
import ModelBaseEcon.getsolverdata
import ModelBaseEcon.hassolverdata
import ModelBaseEcon.setsolverdata!

using ..Plans
import ..SimData

import ..steadystatearray
import ..steadystatedata
import ..StackedTimeSolver

include("firstorder/QZ.jl")
include("firstorder/solve.jl")
include("firstorder/simulate.jl")
include("firstorder/shockdecomp.jl")

end



