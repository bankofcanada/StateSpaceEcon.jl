##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


function rand_factorshocks(fb::ARFactorBlock, len::Integer=1)
    permutedims(rand(MvNormal(ModelBaseEcon._covariance(fb)), len))
end

function rand_observedshocks(m::DFMModel, len::Integer=1)
    permutedims(rand(MvNormal(m.covariance[]), len))
end

"""
    rand_shocks!(data, model::DFMModel, plan)
    rand_shocks!(data, model::DFMModel, range)

Draw random values for the stochastic shocks in the model. The given `data` is
modified in place. The random draws are added to any values that may be in
`data` already.

If plan is given, then the effective range is the simulation portion of the
plan, i.e. random value are not drawn over the initial conditions.

Return `data`.
"""
function rand_shocks! end
export rand_shocks!

function rand_shocks!(data::AbstractMatrix, model::DFMModel, plan::Plan)
    rng = model.maxlag:length(plan)-model.maxlead
    if data isa MVTSeries
        offset = convert(Int, firstdate(plan) - firstdate(data))
        rng = offset .+ rng
    else
        if size(data, 1) != length(plan)
            error("Data and plan sizes don't match")
        end
    end
    rand_shocks!(data, model, rng)
end
rand_shocks!(data::AbstractMatrix, model::DFMModel, rng::AbstractUnitRange{<:MIT}) = error("Invalid use of MIT range with $(typeof(data))")
function rand_shocks!(data::MVTSeries, model::DFMModel, rng::AbstractUnitRange{<:MIT})
    offset = convert(Int, first(rng) - firstdate(data))
    rand_shocks!(data, model, offset .+ (1:length(rng)))
end
function rand_shocks!(data::AbstractMatrix, m::DFMModel, rng::AbstractUnitRange{Int})
    len = length(rng)
    mvars = varshks(m)
    # observed shocks
    begin
        inds = indexin(observedshocks(m), mvars)
        shks = rand_observedshocks(m, len)
        data[rng, inds] .= shks
    end
    # factor shocks
    for fb in ModelBaseEcon._blocks(m)
        inds = indexin(shocks(fb), mvars)
        shks = rand_factorshocks(fb, len)
        data[rng, inds] .= shks
    end
    return data
end

"""
    _simulate_block!(data, fb::ARFactorBlock, range, mvars)

Simulate the given factor block over the given range. `data` is both input and
output - it should contain the shocks and will be updated with the simulated
factors. Input `mvars` holds the model variable names and is used to figure out
which columns of `data` hold the data for the given factor block.
"""
function _simulate_block!(data, fb, rng, f_inds, s_inds)
    for t in rng
        factor = view(data, t, f_inds)
        factor .= data[t, s_inds]
        for lag = 1:fb.order
            factor += fb.arcoefs[lag] * data[t-lag, f_inds]
        end
    end
    return
end

function _simulate_dfm!(data, m, rng)
    mvars = varshks(m)
    for var in observed(m)
        ov_ind = indexin([var], mvars)[1]
        os_ind = indexin([Symbol(var, "_shk")], mvars)[1]
        if os_ind === nothing
            # this observed has no shock (it has an idiosyncratic component)
            data[rng, ov_ind] .= m.mean[ov_ind]
        else
            # this observed has a shock
            data[rng, ov_ind] = m.mean[ov_ind] .+ data[rng, os_ind]
        end
    end
    for fb in ModelBaseEcon._blocks(m)
        f_inds = Int[indexin(factors(fb), mvars)...]
        s_inds = Int[indexin(factorshocks(fb), mvars)...]
        o_inds = Int[indexin(observed(fb), mvars)...]
        _simulate_block!(data, fb, rng, f_inds, s_inds)
        data[rng, o_inds] .+= data[rng, f_inds] * fb.loadings'
    end
    return data
end

function StateSpaceEcon.simulate(m::DFMModel, p::Plan, data::AbstractMatrix; verbose=false)
    sim_rng = 1+m.maxlag:length(p)-m.maxlead
    if data isa MVTSeries
        offset = convert(Int, firstdate(p) - firstdate(data))
        sim_rng = offset .+ sim_rng
    else
        @assert size(data, 1) == length(p)
    end
    verbose && @info "Simulating over $sim_range"
    sim = _simulate_dfm!(copy(data), m, sim_rng)
    return sim
end


#= 
function _simulate!(fb, inds, o_inds, rng, data)
    nf = DFMModels._nfactors(fb)
    @assert 2nf == length(inds)
    f_inds = inds[begin:nf]
    s_inds = inds[nf+1:end]
    factor = Vector{Float64}(undef, nf)
    for t in rng
        fill!(factor, 0)
        for l = 1:DFMModels._order(fb)
            factor += DFMModels._arcoefs(fb, l) * data[t-l, f_inds]
        end
        factor += data[t, s_inds]
        data[t, f_inds] = factor
    end
    data[rng, o_inds] += data[rng, f_inds] * DFMModels._loadings(fb)
    return
end

function StateSpaceEcon.simulate(m::DFMModels.FactorModel, p::Plan, data::AbstractMatrix)
    @info "Simulating $(typeof(m))"
    data = copy(data)
    m_vs = m.varshks
    o_var_inds = indexin(m.variables, m_vs)
    rng = m.maxlag+1:length(p.range)-m.maxlead
    data[rng, o_var_inds] .= reshape(m.mean, 1, :)
    for fb in m.factorblocks
        blk_s_var_inds = convert(Vector{Int}, indexin(fb.stateshks, m_vs))
        blk_o_var_inds = convert(Vector{Int}, indexin(fb.variables, m_vs))
        _simulate!(fb, blk_s_var_inds, blk_o_var_inds, rng, data)
    end
    return data
end
 =#