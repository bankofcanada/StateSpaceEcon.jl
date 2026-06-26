##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# FirstOrderSolver - first-order rational-expectations (Blanchard-Kahn / QZ)
# solver.
#
# The API consumes a compiled model plus a steady-state vector and operates
# on the unified `[vars; shocks]` solver-space array layout shared with
# SimData.
#
# The first-order solve runs entirely on assembled matrices and never
# touches equation kernels.
# ----------------------------------------------------------------------

module FirstOrderSolver

using LinearAlgebra
using SparseArrays

using ModelBaseEcon
using ModelBaseEcon: IR, Symbolic

export FirstOrderModel, first_order_solve
export first_order_simulate
export FirstOrderShockDecompResult, first_order_shockdecomp
export QZResult, run_qz, VarMaps, BlanchardKahnError

include("firstorder/QZ.jl")
include("firstorder/solve.jl")
include("firstorder/simulate.jl")
include("firstorder/shockdecomp.jl")

end # module FirstOrderSolver
