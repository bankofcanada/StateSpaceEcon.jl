##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

module DFMSolver

# Julia Standard Library
using Random
using LinearAlgebra
using Statistics
using SparseArrays

# juliastats.org
using ComponentArrays
using Distributions

# misc 
using UnPack

# github.com/bankofcanada
using TimeSeriesEcon
using ModelBaseEcon
using ModelBaseEcon.DFMModels

using ..StateSpaceEcon
using ..Plans
using ..Kalman

include("dfm/plandata.jl")
include("dfm/random.jl")
include("dfm/simulate.jl")
include("dfm/kalman.jl")
include("dfm/em.jl")

end

using .DFMSolver
export rand_shocks!
export simulate!
