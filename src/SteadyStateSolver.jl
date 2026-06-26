##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

module SteadyStateSolver

using ModelBaseEcon
using ModelBaseEcon: IR, Symbolic
using LinearAlgebra
using LinearAlgebra: norm, lu, qr, ColumnNorm, cond, opnorm

export SteadyStateProblem, sssolve!, ss_residual!, ss_RJ!
export SteadyStateDiagnosis, diagnose_sstate

# Hook overridden by the diagnostics code below. Declared here so
# `sssolve!` can call it without a forward reference; the real method
# lands when `steadystate/diagnose.jl` is included.
function _diagnose_for_sssolve end

include("steadystate/solverdata.jl")
include("steadystate/sssolve.jl")
include("steadystate/diagnose.jl")

end # module SteadyStateSolver
