"""
    StateSpaceEcon

A package for Macroeconomic modelling.

"""
module StateSpaceEcon

using TimeSeriesEcon
using ModelBaseEcon
# using ModelBaseEcon.OptionsMod
# using ModelBaseEcon.Timer

include("SteadyStateSolver.jl")
include("Plans.jl")
include("StackedTimeSolver.jl")
include("misc.jl")

end # module
