##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

export solve!
"""
    solve!(m::Model, [solver::Symbol])

Solve the given model and update its `m.solverdata` according to the specified
solver.  The solver is specified as a `Symbol`.  The default is `solve=:stackedtime`. 

`sover=:firstorder` is still experimental.

"""
function solve!(m::Model, solver::Symbol=:stackedtime)
    return solver === :stackedtime ? m :  # StackedTimeSolver doesn't need a solve! method
           solver === :firstorder ? FirstOrderSolver.solve!(m) :
           throw(ArgumentError("Unknown solver :$solver"))
end
