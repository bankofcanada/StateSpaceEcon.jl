##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    SteadyStateSolver

A module that is part of StateSpaceEcon package. Contains methods for finding a
steady state solution of a model.
"""
module SteadyStateSolver

using SparseArrays
using LinearAlgebra
using ForwardDiff
using DiffResults
using Printf

using ModelBaseEcon

include("steadystate/1dsolvers.jl")
include("steadystate/presolve.jl")
include("steadystate/initial.jl")
include("steadystate/global.jl")
include("steadystate/solverdata.jl")
include("steadystate/nr.jl")
include("steadystate/lm.jl")
include("steadystate/diagnose.jl")
include("steadystate/sssolve.jl")

end # module

using .SteadyStateSolver
export clear_sstate!
export initial_sstate!
export check_sstate
export diagnose_sstate
export sssolve!
