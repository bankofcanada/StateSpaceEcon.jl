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

include("kalman/common.jl")
include("kalman/plain.jl")
include("kalman/sqrt.jl")
include("kalman/smoother.jl")

end
