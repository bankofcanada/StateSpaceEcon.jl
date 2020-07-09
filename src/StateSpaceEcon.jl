"""
    StateSpaceEcon

A package for Macroeconomic modelling.

"""
module StateSpaceEcon

using ModelBaseEcon
using ModelBaseEcon.OptionsMod
using ModelBaseEcon.Timer

include("SteadyStateSolver.jl")
using .SteadyStateSolver

end # module
