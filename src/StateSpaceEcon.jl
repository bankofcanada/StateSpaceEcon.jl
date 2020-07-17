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
include("misc.jl")
include("StackedTimeSolver.jl")

end # module
