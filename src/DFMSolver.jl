##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

module DFMSolver

# Julia Standard Library
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

# juliastats.org
using ComponentArrays
using Distributions

# misc 
using UnPack
using Interpolations
using NaNStatistics

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
