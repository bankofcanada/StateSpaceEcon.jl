"""
    DynamicSolver

A module that is part of StateSpaceEcon package.
Contains methods for solving the dynamic system of equations
for the model and running simulations.
"""
module DynamicSolver

using SparseArrays
using LinearAlgebra
using ForwardDiff
using DiffResults
using Printf

using ModelBaseEcon

include("dynamic/abstract.jl")

end # module

using .DynamicSolver
export FCType, fcgiven, fclevel, fcslope
