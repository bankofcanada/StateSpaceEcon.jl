##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################
# The TimeSeriesEcon-free part of the DFM random surface - the `ShocksSampler`
# (a Distributions.Sampleable over the model shocks) and its constructors /
# `_rand!` / `_scale`. The `rand!`/`rand_shocks!` methods that write into a
# TimeSeriesEcon `SimData`/`MVTSeries` over a calendar range live in a
# TimeSeriesEcon-gated file (loaded only when TimeSeriesEcon is available), since
# this package works in plain matrices and does not mirror the `Plan` API.
##################################################################################

import ModelBaseEcon.DFMModels: SymVec

export ShocksSampler
struct ShocksSampler{M,F} <: Distributions.Sampleable{Multivariate,Continuous}
    names::Vector{Symbol}
    cov::M
    fact::F
end

ShocksSampler(names::SymVec, Sigma::AbstractVecOrMat) = ShocksSampler(Symbol[Symbol(n) for n in names], Sigma)
ShocksSampler(names::Vector{Symbol}, variances::AbstractVector) = ShocksSampler(names, Diagonal(variances))
function ShocksSampler(names::Vector{Symbol}, Sigma::AbstractMatrix)
    isdiag(Sigma) && return ShocksSampler(names, Diagonal(diag(Sigma)))
    issymmetric(Sigma) && return ShocksSampler(names, Symmetric(Sigma))
    # ? should this be an ArgumentError?
    return ShocksSampler(names, Symmetric(0.5 * (Sigma + Sigma')))
end
function ShocksSampler(names::Vector{Symbol}, cov::Diagonal)
    fact = Diagonal(sqrt.(diag(cov)))
    ShocksSampler{typeof(cov),typeof(fact)}(names, cov, fact)
end
function ShocksSampler(names::Vector{Symbol}, cov::Symmetric)
    fact = cholesky(cov)
    ShocksSampler{typeof(cov),typeof(fact)}(names, cov, fact)
end

ShocksSampler(dfm::DFM) = ShocksSampler(dfm.model, dfm.params)
function ShocksSampler(bm::ModelBaseEcon.DFMModels.DFMBlockOrModel, p::DFMParams)
    ShocksSampler(shocks(bm), get_covariance(bm, p))
end

Base.length(s::ShocksSampler) = size(s.cov, 1)
Base.show(io::IO, s::ShocksSampler) = (
    io = IOContext(io, :compact => true, :limit => true);
    println(io, "ShocksSampler");
    println(io, "  shocks: ", join(s.names, ","));
    println(io, "  covariance: ", summary(s.cov));
    Base.print_array(io, s.cov)
)

##################################################################################

function _scale(s::ShocksSampler{<:Diagonal,<:Diagonal}, x::AbstractVecOrMat)
    lmul!(s.fact, x)
end
function _scale(s::ShocksSampler{<:Symmetric,<:Cholesky}, x::DenseVecOrMat)
    # if x is dense, then lmul! works in place with a LowerTriangularMatrix
    lmul!(s.fact.L, x)
end
function _scale(s::ShocksSampler{<:Symmetric,<:Cholesky}, x::AbstractVecOrMat)
    # in the general case, lmul! doesn't work in place, so we need a DenseVecOrMat copy of x
    copyto!(x, lmul!(s.fact.L, copy(x)))
end

function Distributions._rand!(rng::AbstractRNG, s::ShocksSampler, x::AbstractVecOrMat)
    _scale(s, randn!(rng, x))
    return x
end
