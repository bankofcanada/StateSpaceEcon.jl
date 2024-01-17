##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

export ShocksSampler
struct ShocksSampler{M,F} <: Distributions.Sampleable{Multivariate,Continuous}
    names::Vector{Symbol}
    cov::M
    fact::F
end
ShocksSampler(names::Vector{ModelVariable}, mat::AbstractMatrix) = ShocksSampler(map(Symbol, names), mat)
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

Distributions.rand!(rng::AbstractRNG, s::ShocksSampler, data::SimData) = Distributions.rand!(rng, s, :, data)
function Distributions.rand!(rng::AbstractRNG, s::ShocksSampler, range::Union{Colon,AbstractUnitRange{<:MIT}}, data::SimData)
    Distributions.rand!(rng, s, transpose(view(data, range, s.names).values))
    return data
end


##################################################################################


"""
    rand_shocks!(dfm::DFM, plan, data)
    rand_shocks!(dfm::DFM, range, data)

Draw random values for the stochastic shocks in the model. The given `data` is
modified in place. The random draws are written in `data` (columns corresponding
to shocks and the rows corresponding to `range`) overwriting any data that may
be in there already.

If plan is given, then the effective range is the simulation portion of the
plan, i.e. random value are not drawn over the periods for initial and final
conditions.

Return `data`.
"""
rand_shocks!
export rand_shocks!

rand_shocks!(dfm::DFM, args...) = rand_shocks!(Random.default_rng(), dfm, args...)
rand_shocks!(s::ShocksSampler, args...) = rand_shocks!(Random.default_rng(), s, args...)
rand_shocks!(rng::AbstractRNG, dfm::DFM, args...) = rand_shocks!(rng, ShocksSampler(dfm), args...)
rand_shocks!(rng::AbstractRNG, dfm::DFM, plan::Plan, data::MVTSeries) = rand_shocks!(rng, ShocksSampler(dfm), firstdate(plan)+lags(dfm):lastdate(plan)-leads(dfm), data)
function rand_shocks!(rng::AbstractRNG, s::ShocksSampler, range::AbstractUnitRange, data::MVTSeries)
    Distributions.rand!(rng, s, range, data)
end

