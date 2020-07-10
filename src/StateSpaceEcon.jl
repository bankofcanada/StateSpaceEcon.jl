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
export clear_sstate!
export initial_sstate!
export check_sstate
export sssolve!

end # module
