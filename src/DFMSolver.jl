##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

module DFMSolver

# Julia Standard Library
using LinearAlgebra: LinearAlgebra, BLAS, Cholesky, ColumnNorm, Diagonal, I, LAPACK, PosDefException, Symmetric, UpperTriangular,
    cholesky, cholesky!, diag, dot, eigen, isdiag, issymmetric, ldiv!, lmul!, lu, mul!, qr!, rdiv!, rmul!, svd
using Printf: Printf, @sprintf
using Random: Random, AbstractRNG, randn!
using SparseArrays: SparseArrays, blockdiag, nnz, nonzeros, nzrange, rowvals, spzeros
using Statistics: Statistics, mean, mean!

# juliastats.org
using ComponentArrays: ComponentArrays
using Distributions: Distributions, Continuous, Multivariate, cov, params, var

# misc 
using UnPack: UnPack, @unpack
using Interpolations: Interpolations, FritschCarlsonMonotonicInterpolation, interpolate
using NaNStatistics: NaNStatistics, nanmean, nanmedian, nanstd

# github.com/bankofcanada
using TimeSeriesEcon: TimeSeriesEcon, MIT, MVTSeries, Unit, Workspace, Y, firstdate, lastdate
using ModelBaseEcon: ModelBaseEcon, eval_RJ, isshock, shocks
using ModelBaseEcon.DFMModels: DFMModels, DFM, DFMBlock, DFMModel, DFMParams, IdiosyncraticComponents,
    get_covariance, lags, leads, nendog, nobserved, nstates, nstates_with_lags, observed, states_with_lags, varshks

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
