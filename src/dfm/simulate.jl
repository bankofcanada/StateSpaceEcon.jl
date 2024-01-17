##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

import ..simulate
export simulate!

simulate(dfm::DFM, plan::Plan, data::AbstractMatrix; verbose=false) = simulate!(dfm, plan, copy(data); verbose)

function _simulate!(blk::DFMBlock, params::DFMParams, range::AbstractUnitRange, data::MVTSeries, ::Val{:empty_plan})
    vs = varshks(blk)
    colinds = TimeSeriesEcon._colind(data, vs)
    t = Int(first(range) - firstdate(data)) + 1
    trng = collect(Int, (-lags(blk):leads(blk)) .+ t)
    point = view(data.values, trng, colinds)
    R, J = eval_RJ(zero(point), blk, params)
    vars_t = falses(size(point)...)
    vars_t[end, 1:nendog(blk)] .= true
    endomask = vec(vars_t)
    exogmask = .!endomask
    J1 = lu(-J[:, endomask]) # J1 is the identity matrix, so maybe this isn't necessary
    R .= J1 \ R
    J2 = J1 \ Matrix(J[:, exogmask])
    for t = range
        point[endomask] = R + J2 * point[exogmask]
        point.indices[1] .+= 1  # shift view one period forward !!! this is a hack, don't do it !!! 
    end
    return data
end

function simulate!(dfm::DFM, plan::Plan, data::AbstractMatrix; verbose = false)
    m = dfm.model
    p = dfm.params
    shks = Bool[isshock(v) for v in varshks(m)]
    empty_plan = all(plan.exogenous[:,shks]) && !any(plan.exogenous[:,.!shks])
    if !empty_plan
        throw(ArgumentError("Non-empty plan not implemented for DFM models"))
    end
    simrange = firstdate(plan) + lags(m) : lastdate(plan) - leads(m)
    for (name, blk) in m.components
        _simulate!(blk, getproperty(p, name), simrange, data, Val(:empty_plan))
    end
    _simulate!(m.observed_block, getproperty(p, :observed), simrange, data, Val(:empty_plan))
    return data
end
