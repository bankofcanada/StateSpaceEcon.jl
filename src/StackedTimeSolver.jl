##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

module StackedTimeSolver

using ModelBaseEcon
using ModelBaseEcon: IR, Symbolic
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC, sparse, nzrange, rowvals, nonzeros

export StackedTimePlan, stacked_RJ!, stacked_residual!, simulate!

include("stackedtime/solverdata.jl")
include("stackedtime/sparse.jl")
include("stackedtime/simulate.jl")

end # module StackedTimeSolver
