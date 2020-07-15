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

using ..Plans

include("stackedtime/abstract.jl")
include("stackedtime/solverdata.jl")
include("stackedtime/simulate.jl")

end # module

using .StackedTimeSolver
export FCType, fcgiven, fclevel, fcslope
export simulate
