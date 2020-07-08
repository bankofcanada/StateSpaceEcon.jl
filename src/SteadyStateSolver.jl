"""
    SteadyStateSolver

A module that is part of StateSpaceEcon package. Contains methods for finding a steady state of a model.
"""
module SteadyStateSolver

using SparseArrays
using LinearAlgebra
using ForwardDiff
using DiffResults

using ModelBaseEcon

include("steadystate/1dsolvers.jl")
include("steadystate/presolve.jl")
include("steadystate/initial.jl")
include("steadystate/global.jl")
include("steadystate/solverdata.jl")

end # module
