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

using SparseArrays: SparseArrays
using LinearAlgebra: LinearAlgebra, ColumnNorm, I, norm, qr, rank
using ForwardDiff: ForwardDiff
using DiffResults: DiffResults
using Printf: Printf, @sprintf

using ModelBaseEcon: ModelBaseEcon, Model, Options, SteadyStateData, SteadyStateEquation, alleqns, geteqn,
    isexog, isshock, issteady, neqns

# World age: a model's evaluation functions are created by `eval` when the model
# (or a steady state constraint) is built. A caller that builds and then solves in
# the same call (a function, a `let` or a `@testset` block) runs in a world where
# those functions do not exist yet. So each public entry point below calls its
# implementation through `Base.invokelatest`, once; the solvers themselves don't.

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
