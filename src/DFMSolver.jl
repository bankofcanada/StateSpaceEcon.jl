##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

module DFMSolver

# Julia Standard Library
using Random
using LinearAlgebra

# juliastats.org
using ComponentArrays
using Distributions

# github.com/bankofcanada
using TimeSeriesEcon
using ModelBaseEcon
using ModelBaseEcon.DFMModels

using ..StateSpaceEcon
using ..Plans
import ..Kalman

include("dfm/plandata.jl")
include("dfm/random.jl")
# include("dfm/simulate.jl")
# include("dfm/kalman.jl")

end

using .DFMSolver
export rand_shocks!

