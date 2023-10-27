##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

module DFMSolver

using LinearAlgebra
using Distributions

using TimeSeriesEcon
using ModelBaseEcon
using ModelBaseEcon.DFMModels

using ..StateSpaceEcon
using ..Plans
import ..Kalman


include("dfm/plandata.jl")
# include("dfm/simulate.jl")
# include("dfm/kalman.jl")

end

using .DFMSolver
export rand_shocks!

