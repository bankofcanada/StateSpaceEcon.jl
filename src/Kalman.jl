##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


module Kalman

using LinearAlgebra

using TimeSeriesEcon
using ModelBaseEcon
using ..StateSpaceEcon

using UnPack
using MacroTools

include("kalman/api.jl")
include("kalman/data.jl")
include("kalman/filter.jl")

# include("kalman/plain.jl")
# include("kalman/sqrt.jl")

include("kalman/smoother.jl")
include("kalman/linear_stationary.jl")

end
