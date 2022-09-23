##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


function Plans.Plan(m::DFMModel, r::AbstractUnitRange)
    if !<:(eltype(r),MIT)
        r = convert(UnitRange{MIT{Unit}}, r)
    end
    range = first(r)-m.maxlag:last(r)+m.maxlead
    names = m.varshks
    nn = length(names)
    p = Plan(range, (; (@. Symbol(names) => 1:nn)...), falses(length(range), nn))
    p.exogenous[:, isshock.(names)] .= true
    return p
end

StateSpaceEcon.zeroarray(m::DFMModel, p::Plan) = zeros(size(p.exogenous))
function StateSpaceEcon.zerodata(m::DFMModel, p::Plan)
    return MVTSeries(p.range, m.varshks, zeros)
end
StateSpaceEcon.zeroworkspace(m::DFMModel, p::Plan) = Workspace(
    v => zeros(p.range) for v in m.varshks
)

StateSpaceEcon.steadystatearray(m::DFMModel, p::Plan) = steadystatedata(m, p).values
function StateSpaceEcon.steadystatedata(m::DFMModel, p::Plan)
    data = zerodata(m, p)
    data[:, m.variables] .= reshape(m.mean, 1, :)
    return data
end
function StateSpaceEcon.steadystateworkspace(m::DFMModel, p::Plan)
    ret = zeroworkspace(m, p)
    for (i, v) in enumerate(observed(m))
        fill!(ret[v], m.mean[i])
    end
    return ret
end

