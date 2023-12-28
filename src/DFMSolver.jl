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
using Statistics

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

# include("dfm/kalman_old.jl")   ## -- the old stuff
# include("dfm/kalman_blk.jl")   ## -- the new stuff
include("dfm/kalman.jl")   ## -- the new new stuff

end

using .DFMSolver
export rand_shocks!
export simulate!
