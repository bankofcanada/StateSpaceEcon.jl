##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

# This file contains the main interface of the simulate() function
# It processes user inputs and dispatches to the appropriate
# specialization.  Each solver should have its own version of simulate().

# same for solve!() and shockdecomp()


const defaultsolver = :stackedtime

#####  solve!() #####

"""
    solve!(m::Model; [solver::Symbol])

Solve the given model and update its `m.solverdata` according to the specified
solver.  The solver is specified as a `Symbol`.  The default is `solve=:stackedtime`.
"""
function solve! end
export solve!

function solve!(model::Model; solver::Symbol=defaultsolver, kwargs...)
    return getsolvermodule(solver).solve!(model; kwargs...)
end

#####  shockdecomp() #####

"""
    shockdecomp(model, plan, exog_data; control, solver, [options])

Compute the shock decomposition for the given model, plan, exogenous (shocks)
data and control solution.

If the `control` option is not specified we use the steady state solution stored
in the model instance. The algorithm assumes that `control` is a solution to the
dynamic model for the given plan range and final condition. We verify the
residual and issue a warning, but do not enforce this. See
[`steadystatedata`](@ref).

As part of the algorithm we run a simulation with the given plan, data and solver. 
See [`simulate`](@ref) for additional options that control the simulation.

Note that unlike [`simulate`](@ref), here we require `exogdata` and `control` to
be `MVTSeries`, i.e. plain `Array` or `Workspace` are not allowed. 

The returned value is a `Workspace` with three things in it: `c` contains a copy
of `control`, `s` contains the simulated solution for the given `plan` and
`exogdata` and `sd` contains the shock decomposition data. The data in `sd` is
organized as a `Workspace` containing an `MVTSeries` for each variable,

"""
function shockdecomp end
export shockdecomp

function shockdecomp(model::Model, plan::Plan, exogdata::MVTSeries;
    solver::Symbol=defaultsolver, initdecomp::Workspace=Workspace(),
    control::MVTSeries=steadystatedata(model, plan), kwargs...)
    return getsolvermodule(solver).shockdecomp(model, plan, exogdata; control, initdecomp, kwargs...)
end

#####  simulate() #####

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
    * `solver::Symbol` - specify the simulation solver. Available options are
      :stackedtime and :firstorder. If not given, default is :stackedtime.
    * `verbose::Bool` - control whether or not to print progress information.
        Default value is taken from `model.options`.
    * `tol::Float64` - set the desired accuracy. Default value is taken from
        `model.options`.
    * `maxiter::Int` - algorithm fails if the desired accuracy is not reached
        within this maximum number of iterations. Default value is taken from
        `model.options`.
The following options are specific to the `:stackedtime` solver
    * `sim_solver` - specify the non-linear solver to use. Available options are 
      - `:sim_nr` : (default) Newton-Raphson, with possible damping, see below.
      - `:sim_lm` : Levenberg–Marquardt
      - `:sim_gn` : Gauss-Newton
    * `linesearch::Bool` - When `true` the Newton-Raphson is modified to include
      a search along the descent direction for a sufficient decrease in f. It
      will do this at each iteration. Default is `false`. (Superseded by the
      `damping` option described below)
    * `damping` - Specifies the style of damping that can be applied to the
      Newton non-linear solver. Available options are:
      - if not given the default behaviour is no damping, i.e. the damping
        coefficient is set to 1.0 in each iteration.
      - number: the damping coefficient will be set to the given number (rather than 1)
      - vector of numbers: the damping coefficient in each iteration will be set
        the number in the corresponding entry of the given vector. If there are
        more Newton iterations than the length of the vector, the last entry
        will be used until in the remaining iterations.
      - `:linesearch` or `:armijo` : same as setting `linesearch=true`. The
        Armijo rule is taken from "C.T.Kelly, Iterative Methods for Linear and
        Nonlinear Equations, ch.8.1, p.137"
      - `(:armijo, :sigma => 0.5, :alpha => 1e-4)` - override the default
        parameters of the Armijo rule.
      - `:br81` : (experimental) implements the damping algorithm in "Bank, R.E.,
        Rose, D.J. Global approximate Newton methods. Numer. Math. 37, 279–295
        (1981)."
      - `(:br81, :rateK => 10, :delta => 0.1)` : override the default parameters
        of the Bank & Rose (1981) algorithm.

"""
function simulate end
export simulate

# The versions of simulate with Dict/Workspace -> convert to SimData
simulate(m::Model, p::Plan, data::Union{Workspace,AbstractDict}; kwargs...) =
    simulate(m, p, workspace2data(TimeSeriesEcon._dict_to_workspace(data), m, p; copy=true); kwargs...)

simulate(m::Model, p_ant::Plan, data_ant::Union{Workspace,AbstractDict},
    p_unant::Plan, data_unant::Union{Workspace,AbstractDict}; kwargs...) =
    simulate(m, p_ant, workspace2data(TimeSeriesEcon._dict_to_workspace(data_ant), m, p_ant; copy=true),
        p_unant, workspace2data(TimeSeriesEcon._dict_to_workspace(data_unant), m, p_unant; copy=true), ;
        kwargs...)

function _initial_guess_to_array(initial_guess, m, p)
    return initial_guess isa SimData ? (; initial_guess=data2array(initial_guess, m, p)) :
           initial_guess isa Workspace ? (; initial_guess=workspace2array(initial_guess, m, p)) :
           initial_guess isa AbstractDict ? (; initial_guess=workspace2array(Workspace(initial_guess), m, p)) :
           (;)
end

# Handle initial conditions and assign result only within the plan range (in case range of given data is larger)
function simulate(m::Model, p::Plan, data::SimData; kwargs...)
    exog = data2array(data, m, p)
    kw_ig = _initial_guess_to_array(get(kwargs, :initial_guess, nothing), m, p)
    result = copy(data)
    result[p.range, m.varshks] .= simulate(m, p, exog; kwargs..., kw_ig...)
    return result
end

# this is the dispatcher -> call the appropriate solver
simulate(m::Model, p::Plan, exog::AbstractMatrix; solver::Symbol=defaultsolver, kwargs...) =
    getsolvermodule(solver).simulate(m, p, exog; kwargs...)

# Handle the case with 2 sets of plan-data for mixture of ant and unant shocks
function simulate(m::Model, p_ant::Plan, data_ant::SimData, p_unant::Plan, data_unant::SimData; kwargs...)
    exog_ant = data2array(data_ant)
    exog_unant = data2array(data_unant)
    kw_ig = _initial_guess_to_array(get(kwargs, :initial_guess, nothing), m, p_ant)
    result = copy(data_ant)
    result[p_ant.range, m.varshks] .= simulate(m, p_ant, exog_ant, p_unant, exog_unant; kwargs..., kw_ig...)
    return result
end

# this is the dispatcher -> call the appropriate solver
simulate(m::Model, p_ant::Plan, data_ant::AbstractMatrix,
    p_unant::Plan, data_unant::AbstractMatrix;
    solver::Symbol=defaultsolver, kwargs...) =
    getsolvermodule(solver).simulate(m, p_ant, data_ant, p_unant, data_unant; kwargs...)


#####  stoch_simulate() #####

"""
    stoch_simulate(model, plan, baseline, shocks; control, ...)

Run multiple simulations with the given shocks centered about the given control
(steady state by default).

The baseline should span the plan range and must be given in levels ( i.e., option deviation=true is not implemented)

The shocks can be given as a collection of random realizations, where each
realization could be an MVTSeries or a Workspace. Only the shocks should be
provided.

All shock names must be exogenous in the given plan over the range of the given
data. The data in `control` is taken as anticipated and the stochastic
realizations are taken as unanticipated and the same plan is used for both
components.

Currently only the case of `solver=stackedtime` is implemented.

"""
function stoch_simulate end


######## make sure the baseline argument is a SimData

# convert baseline from Workspace to SimData
function stoch_simulate(m::Model, p::Plan, baseline::Workspace, shocks; kwargs...)
    baseline = copyto!(MVTSeries(p.range, m.allvars), baseline)
    return stoch_simulate(m, p, baseline, shocks; kwargs...)
end

# convert baseline from Matrix to SimData
function stoch_simulate(m::Model, p::Plan, baseline::AbstractMatrix, shocks; kwargs...)
    baseline = copyto!(MVTSeries(p.range, m.allvars), baseline)
    return stoch_simulate(m, p, baseline, shocks; kwargs...)
end

######## dispatcher
function stoch_simulate(m::Model, p::Plan, baseline::SimData, shocks;
    solver::Symbol=defaultsolver,
    kwargs...
)
    ##### check shocks
    if shocks isa SimData && eltype(shocks) == Float64
        shocks = [shocks]
    end
    # are shocks of the appropriate type
    if ((shocks isa AbstractVector) && (eltype(shocks) <: SimData)) ||
       ((shocks isa Workspace) && (Base.promote_typeof(values(shocks)...) <: SimData))
        nothing
    else
        throw(ArgumentError("Expected the shocks argument to be Vector or Workspace of SimData, not $(typeof(shocks))."))
    end
    ##### dispatch
    getsolvermodule(solver).stoch_simulate(m, p, baseline[p.range], shocks; kwargs...)
end
export stoch_simulate

