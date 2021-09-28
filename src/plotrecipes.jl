##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

using Plots
@recipe plot(sd::SimData...; vars=nothing, names=nothing) = begin
    if vars === nothing
        error("Must specify variables to plot")
    elseif vars == :all
        vars = StateSpaceEcon._names(sd[1])
    end
    if length(vars) > 10
        error("Too many variables. Split into pages.")
    end
    if names === nothing
        names = ["data$i" for i = 1:length(sd)]
    end
    layout --> length(vars)
    title --> reshape(map(string, [vars...]), 1, :)
    titlefont --> font(10, :bold)
    label --> repeat(reshape([names...], 1, :), inner=(1, length(vars)))
    linewidth --> 1.5
    for linesym in (:linestyle, :linewidth, :linecolor)
        val = get(plotattributes, linesym, nothing)
        if val !== nothing && length(val) == length(names)
            plotattributes[linesym] = 
                repeat(reshape([val...], 1, :), inner=(1, length(vars)))
        end
    end
    left_margin --> 4 * Plots.mm
    for s in sd
        for v in vars
            @series begin
                s[v]
            end
        end
    end
end
