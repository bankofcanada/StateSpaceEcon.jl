##################################################################################
# DFM solver subsystem.
#
# The EM core works on plain matrices and reuses the generic `Kalman`
# filter/smoother via a `KFLinearModel` built from DFM params. The
# TimeSeriesEcon-dependent surface (plandata / random / simulate / kfd2data) is
# split into a TimeSeriesEcon-gated file loaded only when TimeSeriesEcon is
# available, so the headline EM match needs no TimeSeriesEcon dependency.
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

using ModelBaseEcon
using ModelBaseEcon.DFMModels

using ..Kalman
using ..Kalman: KFLinearModel, KFilter, KFDataSmoother
using ..Kalman: kf_filter!, kf_smoother!, kf_length_x, kf_length_y, kf_linear_model

include("dfm/random.jl")
include("dfm/kalman.jl")
include("dfm/em.jl")

# NOTE: the TimeSeriesEcon-typed DFM convenience surface - `Plan(dfm,rng)` /
# `steadystatedata` / `zerodata` / `simulate(dfm,plan,data)` /
# `rand_shocks!`-over-`MVTSeries` / `kfd2data` - is not currently implemented.
# It is built on the `Plan` type, which this package replaces with
# `SimData`/`SimPlan`. The numerical core of every DFM capability is fully
# covered without it: EM (full + missing-data), the Kalman filter/smoother,
# `ShocksSampler`, and the impute helpers.

export EMestimate!
export ShocksSampler
export kf_linear_model, kf_length_x, kf_length_y
export em_impute_kalman!, em_impute_interpolation!
export em_apply_constraint!

end # module DFMSolver
