##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


#############################################################
#  API - if you want to use Kalman, your model must define 
#    methods for your model type for these functions

nobserved(model) = error("Method not defined for models of type $(typeof(model))")
nobservedshocks(model) = error("Method not defined for models of type $(typeof(model))")
nstates(model) = error("Method not defined for models of type $(typeof(model))")
nstateshocks(model) = error("Method not defined for models of type $(typeof(model))")

eval_transition(model, state::AbstractVector{Float64}, stateshock::AbstractVector{Float64}, period::Integer) = error("Method not defined for models of type $(typeof(model))")
stateshocks_covariance(model, period::Integer) = error("Method not defined for models of type $(typeof(model))")
eval_observation(model, state::AbstractVector{Float64}, observedshock::AbstractVector{Float64}, period::Integer) = error("Method not defined for models of type $(typeof(model))")
observedshocks_covariance(model, period::Integer) = error("Method not defined for models of type $(typeof(model))")

export nobserved, nobservedshocks,
    nstates, nstateshocks,
    eval_transition, eval_observation,
    stateshocks_covariance, observedshocks_covariance

############################################################

# Common base type for Kalman data.
# Sizes are type parameters, so we don't have to check if they match all the time.
#  NO = number of observations
#  NS = number of states 
#  NP = number of periods 
abstract type KalmanData{NO,NS,NP} end

"""
    kdstore!(kd::KalmanData, period::Integer, name::Symbol, value)

Assign the value for the given name and period within the [`KalmanData`](@ref) `kd`.

If you implement your custom `KalmanData` type, you must provide a method of this function
for your data type.
"""
kdstore!(kd::KalmanData, period::Integer, name::Symbol, value) = error("Not implemented for $(typeof(kd)).")

abstract type KalmanPlainData{NO,NS,NP} <: KalmanData{NO,NS,NP} end
function kdstore!(kd::KalmanPlainData, period::Integer, name::Symbol, value)
    if hasproperty(kd, name)
        setindex!(getproperty(kd, name), value, period, repeat([:], ndims(value))...)
    else
        return value
    end
end
function kdstore!(kd::KalmanPlainData, period::Integer; kwargs...)
    for (name, value) in kwargs
        kdstore!(kd, period, name, value)
    end
    return
end
function kdfetch(kd::KalmanPlainData, period::Integer, name::Symbol)
    if hasproperty(kd, name)
        prop = getproperty(kd, name)
        return getindex!(prop, period, repeat([:], ndims(prop) - 1))
    end
    return getproperty(kd, name)  ## throws an error
end

# # "plain" Kalman filter is the regular one
# # In "sqrt" version the covariance matrices are expressed in terms of 
# #    their cholesky decompositions.
# const KalmanMethods = Union{Val{:plain}, Val{:sqrt}}

struct KalmanFullData{NO,NS,NP} <: KalmanPlainData{NO,NS,NP}
    x::Array{Float64,2}
    xpred::Array{Float64,2}
    xsmooth::Array{Float64,2}
    ypred::Array{Float64,2}
    ysmooth::Array{Float64,2}
    Px::Array{Float64,3}
    Pxpred::Array{Float64,3}
    Pxsmooth::Array{Float64,3}
    Pypred::Array{Float64,3}
    Pysmooth::Array{Float64,3}
    Pxypred::Array{Float64,3}
    K::Array{Float64,3}
    J::Array{Float64,3}
    log_lik::Vector{Float64}
    sqr_res::Vector{Float64}
    KalmanFullData(nobserved::Integer, nstates::Integer, nperiods::Integer) = new{nobserved,nstates,nperiods}(
        Array{Float64}(undef, nperiods, nstates),
        Array{Float64}(undef, nperiods, nstates),
        Array{Float64}(undef, nperiods, nstates),
        Array{Float64}(undef, nperiods, nobserved),
        Array{Float64}(undef, nperiods, nobserved),
        Array{Float64}(undef, nperiods, nstates, nstates),
        Array{Float64}(undef, nperiods, nstates, nstates),
        Array{Float64}(undef, nperiods, nstates, nstates),
        Array{Float64}(undef, nperiods, nobserved, nobserved),
        Array{Float64}(undef, nperiods, nobserved, nobserved),
        Array{Float64}(undef, nperiods, nstates, nobserved),
        Array{Float64}(undef, nperiods, nstates, nobserved),
        Array{Float64}(undef, nperiods, nstates, nstates),  # J
        fill(-Inf, nperiods),
        fill(Inf, nperiods),
    )
end

# use this when the smoother isn't being run
struct KalmanFilterOnlyData{NO,NS,NP} <: KalmanPlainData{NO,NS,NP}
    x::Array{Float64,2}
    xpred::Array{Float64,2}
    ypred::Array{Float64,2}
    Px::Array{Float64,3}
    Pxpred::Array{Float64,3}
    Pypred::Array{Float64,3}
    Pxypred::Array{Float64,3}
    K::Array{Float64,3}
    log_lik::Vector{Float64}
    sqr_res::Vector{Float64}
    KalmanFilterOnlyData(nobserved::Integer, nstates::Integer, nperiods::Integer) = new{nobserved,nstates,nperiods}(
        Array{Float64}(undef, nperiods, nstates),
        Array{Float64}(undef, nperiods, nstates),
        Array{Float64}(undef, nperiods, nobserved),
        Array{Float64}(undef, nperiods, nstates, nstates),
        Array{Float64}(undef, nperiods, nstates, nstates),
        Array{Float64}(undef, nperiods, nobserved, nobserved),
        Array{Float64}(undef, nperiods, nstates, nobserved),
        Array{Float64}(undef, nperiods, nstates, nobserved),
        fill(-Inf, nperiods),
        fill(Inf, nperiods),
    )
end

# Use for ML estimation
struct KalmanLikelihoodData{NO,NS,NP} <: KalmanPlainData{NO,NS,NP}
    log_lik::Vector{Float64}
    KalmanLikelihoodData(nobserved::Integer, nstates::Integer, nperiods::Integer) = new{nobserved,nstates,nperiods}(
        fill(-Inf, nperiods)
    )
end

# Use for least squares estimation
struct KalmanSquaredResidualData{NO,NS,NP} <: KalmanPlainData{NO,NS,NP}
    sqr_res::Vector{Float64}
    KalmanSquaredResidualData(nobserved::Integer, nstates::Integer, nperiods::Integer) = new{nobserved,nstates,nperiods}(
        fill(-Inf, nperiods)
    )
end

kalman_data(model, nperiods::Integer, kind::Symbol=:full, method::Symbol=:plain) =
    method === :plain ? (
        kind === :full ? KalmanFullData(nobserved(model), nstates(model), nperiods) :
        kind === :filter ? KalmanFilterOnlyData(nobserved(model), nstates(model), nperiods) :
        kind === :likelihood ? KalmanLikelihoodData(nobserved(model), nstates(model), nperiods) :
        kind === :residual ? KalmanSquaredResidualData(nobserved(model), nstates(model), nperiods) :
        error("Invalid kind of Kalman data: $kind")
    ) : error("Invalid Kalman method: $method.")


# struct KDSqrtData{NO,NS,NP} <: KalmanData{NO,NS,NP}
#     x::Array{Float64,2}
#     xpred::Array{Float64,2}
#     ypred::Array{Float64,2}
#     Px::Vector{Cholesky{Float64,Matrix{Float64}}}
#     Pxpred::Vector{Cholesky{Float64,Matrix{Float64}}}
#     Pypred::Vector{Cholesky{Float64,Matrix{Float64}}}
#     Pxypred::Vector{Matrix{Float64}}
#     K::Vector{Matrix{Float64}}
#     log_lik::Vector{Float64}
#     KDPlainData(nobserved::Integer, nstates::Integer, nperiods::Integer) = new{nobserved,nstates,nperiods}(
#         Array{Float64}(undef, nperiods, nstates),
#         Array{Float64}(undef, nperiods, nstates),
#         Array{Float64}(undef, nperiods, nobserved),
#         [Cholesky(Matrix{Float64}(undef, nstates, nstates), 'U', 0) for _ in 1:nperiods],
#         [Cholesky(Matrix{Float64}(undef, nstates, nstates), 'U', 0) for _ in 1:nperiods],
#         [Cholesky(Matrix{Float64}(undef, nobserved, nobserved), 'U', 0) for _ in 1:nperiods],
#         [Matrix{Float64}(undef, nstates, nobserved) for _ in 1:nperiods],
#         [Matrix{Float64}(undef, nstates, nobserved) for _ in 1:nperiods],
#         fill(-Inf, nperiods)
#     )
# end

# kalman_data(model, nperiods::Integer, ::Val{:plain}=Val(:plain)) = 
#     KDPlainData(nobserved(model), nstates(model), nperiods)
# kalman_data(model, nperiods::Integer, ::Val{:sqrt}) = 
#     KDSqrtData(nobserved(model), nstates(model), nperiods)
# export kalman_data

