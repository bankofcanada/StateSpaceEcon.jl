##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

# This file contains the main interface of the simulate() function 
# It processes user inputs and dispatches to the appropriate 
# specialization.  Each solver should have its own version of simulate().


export simulate
"""
    simulate(model, plan, data; <options>)

Run a simulation for the given model, simulation plan and exogenous data.

### Arguments
  * `model` - the [`Model`](@ref ModelBaseEcon.Model) instance to simulate.
  * `plan` - the [`Plan`](@ref) for the simulation.
  * `data` - a 2D `Array` containing the exogenous data. This includes the
    initial and final conditions.

### Options as keyword arguments
  * `fctype::`[`FinalCondition`](@ref) - set the desired final condition type
    for the simulation. The default value is [`fcgiven`](@ref). Other possible
    values include [`fclevel`](@ref), [`fcslope`](@ref) and
    [`fcnatural`](@ref).
  * `initial_guess::AbstractMatrix{Float64}` - a 2D `Array` containing the
    initial guess for the solution. This is used to start the Newton-Raphson
    algorithm. The default value is an empty array (`zeros(0,0)`), in which case
    we use the exogenous data for the initial condition. You can use the steady
    state solution using [`steadystatearray`](@ref).
  * `deviation::Bool` - set to `true` if the `data` is given in deviations from
    the steady state. In this case the simulation result is also returned as a
    deviation from the steady state. Default value is `false`.
  * `anticipate::Bool` - set to `false` to instruct the solver that all shocks
    are unanticipated by the agents. Default value is `true`.
  * `verbose::Bool` - control whether or not to print progress information.
    Default value is taken from `model.options`.
  * `tol::Float64` - set the desired accuracy. Default value is taken from
    `model.options`.
  * `maxiter::Int` - algorithm fails if the desired accuracy is not reached
    within this maximum number of iterations. Default value is taken from
    `model.options`.
  * `linesearch::Bool` - When `true` the Newton-Raphson is modified to include a 
    search along the descent direction for a sufficient decrease in f. It will 
    do this at each iteration. Default is `false`.
"""
function simulate end


# The versions of simulate with Dict/Workspace -> convert to SimData
simulate(m::Model, p::Plan, data::AbstractDict; kwargs...) = simulate(m, p, dict2data(data, m, p; copy=true); kwargs...)
simulate(m::Model, p::Plan, data::Workspace; kwargs...) = simulate(m, p, workspace2data(data, m, p; copy=true); kwargs...)

function _initial_guess_to_array(initial_guess, m, p)
    return initial_guess isa SimData ? (; initial_guess=data2array(initial_guess, m, p)) :
           initial_guess isa Workspace ? (; initial_guess=workspace2array(initial_guess, m, p)) :
           initial_guess isa AbstractDict ? (; initial_guess=workspace2array(Workspace(initial_guess), m, p)) :
           (;)
end

# Handle initial conditions and assign result only within the plan range (in case range of given data is larger)
function simulate(m::Model, p::Plan, data::SimData, ; kwargs...)
    exog = data2array(data, m, p)
    kw_ig = _initial_guess_to_array(get(kwargs, :initial_guess, nothing), m, p)
    result = copy(data)
    result[p.range, m.varshks] .= simulate(m, p, exog; kwargs..., kw_ig...)
    return result
end

# this is the dispatcher -> call the appropriate solver
simulate(m::Model, p::Plan, exog::AbstractMatrix; kwargs...) = StackedTimeSolver.simulate(m, p, exog; kwargs...)

# Handle the case with 2 sets of plan-data for mixture of ant and unant shocks
function simulate(m::Model, p_ant::Plan, data_ant::SimData, p_unant::Plan, data_unant::SimData; kwargs...)
    exog_ant = data2array(data_ant)
    exog_unant = data2array(data_unant)
    kw_ig = _initial_guess_to_array(get(kwargs, :initial_guess, nothing), m, p_ant)
    result = copy(data_ant)
    result[p_ant.range, m.varshks] .= simulate(m, p_ant, exog_ant, p_unant, exog_unant; kwargs..., kw_ig...)
    return result
end

simulate(m::Model, p_ant::Plan, data_ant::AbstractMatrix, p_unant::Plan, data_unant::AbstractMatrix; kwargs...) = StackedTimeSolver.simulate(m, p_ant, data_ant, p_unant, data_unant; kwargs...)
