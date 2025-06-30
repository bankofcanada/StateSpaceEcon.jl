##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

import ..Plan
import ..zerodata
import ..zeroarray
import ..zeroworkspace
import ..steadystatedata
import ..steadystatearray
import ..steadystateworkspace

Plan(dfm::DFM, rng) = Plan(dfm.model, rng)
function Plan(m::DFMModel, rng::AbstractUnitRange)
    eltype(rng) <: MIT || (rng = convert(UnitRange{MIT{Unit}}, rng))
    range = first(rng)-lags(m):last(rng)+leads(m)
    names = varshks(m)
    nn = length(names)
    p = Plan(range, (; (@. Symbol(names) => 1:nn)...), falses(length(range), nn))
    p.exogenous[:, isshock.(names)] .= true
    return p
end

@inline zeroarray(::DFM, p::Plan) = zeros(size(p.exogenous))
@inline zeroarray(::DFMModel, p::Plan) = zeros(size(p.exogenous))
@inline zerodata(dfm::DFM, p::Plan) = zerodata(dfm.model, p)
@inline zerodata(m::DFMModel, p::Plan) = MVTSeries(p.range, varshks(m), zeros)
@inline zeroworkspace(dfm::DFM, p::Plan) = zeroworkspace(dfm.model, p)
@inline zeroworkspace(m::DFMModel, p::Plan) = Workspace(v => zeros(p.range) for v in varshks(m))

@inline steadystatedata(dfm::DFM, p::Plan; ref=0) = steadystatedata(dfm.model, p::Plan, dfm.params)
function steadystatedata(m::DFMModel, p::Plan, params::DFMParams; ref=0)
    data = zerodata(m, p)
    # all means are 0 except possibly observed variables
    for on in keys(m.observed)
        ss = params[on].mean
        for var in keys(ss)
            data[:, var] .= ss[var]
        end
    end
    return data
end
@inline steadystatearray(dfm::DFM, p::Plan; ref=0) = steadystatearray(dfm.model, p, dfm.params)
function steadystatearray(m::DFMModel, p::Plan, params::DFMParams; ref=0)
    data = zeroarray(m, p)
    v2i = p.varshks
    for on in keys(m.observed)
        ss = params[on].mean
        for vn in keys(ss)
            data[:, v2i[vn]] .= ss[vn]
        end
    end
    return data
end

@inline steadystateworkspace(m::DFM, p::Plan; ref=0) = steadystateworkspace(m.model, p, m.params)
function steadystateworkspace(m::DFMModel, p::Plan, params::DFMParams; ref=0)
    wks = zeroworkspace(m, p)
    for on in keys(m.observed)
        ss = params[on].mean
        for var in keys(ss)
            fill!(wks[var], ss[var])
        end
    end
    return wks
end

