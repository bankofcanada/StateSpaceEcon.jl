"""
    MTKExt

A module that is part of StateSpaceEcon.jl.
Declares functions for creating ModelingToolkit systems out of `Model`s.
By default, these functions have no methods;
the relevant methods will be added when ModelingToolkit.jl is loaded.
"""
module MTKExt

const MTK_NEEDED_STRING = """
!!! note
    This method requires ModelingToolkit.jl to be loaded
    and only works with Julia 1.9.0 or later.
"""

"""
    stacked_time_system(m::Model, exog_data::Matrix; fctype = fcgiven)

Convert a `Model` into a `ModelingToolkit.NonlinearSystem`
that incorporates the stacked time algorithm.

$MTK_NEEDED_STRING

# Inputs
- `m::Model`: Model to convert.
- `exog_data::AbstractArray{Float64,2}`: Data matrix of size `(NT, nvars + nshks)`,
  where `NT` is the number of simulation periods to simulate (plus lags and leads),
  and `nvars` and `nshks` are the number of variables and shocks in the model.
  This data is used to specify the exogenous data, initial conditions, and
  (if applicable) the final conditions.

# Options
- `fctype = fcgiven`: The class of final conditions to use in the simulation.
  The default is [`fcgiven`](@ref StateSpaceEcon.StackedTimeSolver.fcgiven).

!!! note
    If `fctype` is [`fclevel`](@ref StateSpaceEcon.StackedTimeSolver.fclevel)
    or [`fcslope`](@ref StateSpaceEcon.StackedTimeSolver.fcslope),
    `m` will need its steady state solved prior to calling this function.
    See [`sssolve!`](@ref StateSpaceEcon.SteadyStateSolver.sssolve!) or [`solve_steady_state!`](@ref).

# Example
This function is used to bring a `Model` into
the ModelingToolkit/SciML ecosystem.
Here is an example of calling this function
and then converting the returned system into a `ModelingToolkit.NonlinearProblem`
to be solved with one of the solvers from [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl):
```julia
using ModelBaseEcon, StateSpaceEcon, ModelingToolkit, NonlinearSolve
@using_example E3
m = E3.newmodel()
exog_data = rand(102, 6) # Replace with desired data.
s = create_system(m, exog_data)
nf = NonlinearFunction(s)
u0 = zeros(length(unknowns(s)))
prob = NonlinearProblem(nf, u0)
solver = NewtonRaphson() # Replace with desired solver.
sol = solve(prob, solver)
```

See the [ModelingToolkit.jl docs](https://docs.sciml.ai/ModelingToolkit/stable/tutorials/ode_modeling/)
to see what one can do with the solution object `sol`.

---

    stacked_time_system(sd::StackedTimeSolverData, data::Matrix)

Alternative call signature when `sd` is already available (e.g., created manually).
"""
function stacked_time_system end
export stacked_time_system

"""
    compute_residuals_stacked_time(u, sd::StackedTimeSolverData, data::Matrix)

Compute the residuals of the stacked time system in `sd`
given variable values `u` and exogenous data `data`.
This function is closed over in [`stacked_time_system`](@ref)
to create a function that can be passed to `ModelingToolkit.NonlinearProblem`.

$MTK_NEEDED_STRING

!!! warning
    Internal function not part of the public interface.
"""
function compute_residuals_stacked_time end

"""
    _create_system(prob::ModelingToolkit.NonlinearProblem, sd::StackedTimeSolverData)

Create a `ModelingToolkit.NonlinearSystem` from the given problem.
The solver data `sd` should be the same as used for creating `prob`.

$MTK_NEEDED_STRING

!!! warning
    Internal function not part of the public interface.
"""
function _create_system end

"""
    rename_variables(old_sys::ModelingToolkit.NonlinearSystem, sd::StackedTimeSolverData)

Create a new `ModelingToolkit.NonlinearSystem` by replacing the variable names in `old_sys`
with variable names from the solver data `sd`.
The solver data `sd` should be the same as used for creating `old_sys`.

$MTK_NEEDED_STRING

!!! warning
    Internal function not part of the public interface.
"""
function rename_variables end

"""
    get_var_names(sd::StackedTimeSolverData)

Return the names of endogenous variables and/or shocks that are solved for.
A variable/shock name is returned if the corresponding variable/shock
is included in `sd.solve_mask` for at least one simulation period.

$MTK_NEEDED_STRING

!!! warning
    Internal function not part of the public interface.
"""
function get_var_names end

"""
    solve_steady_state!(m::Model, sys::ModelingToolkit.NonlinearSystem; u0 = zeros(...), solver = nothing, solve_kwargs...)

Solve the steady state system `sys` (created with [`steady_state_system`](@ref))
and store the result in `m`.
The model `m` should be the same model used to create `sys`.

!!! note
    This function is a replacement for [`sssolve!`](@ref StateSpaceEcon.SteadyStateSolver.sssolve!)
    and uses the ModelingToolkit ecosystem for solving.

$MTK_NEEDED_STRING

# Inputs
- `m::Model`: Model whose steady state should be solved.
- `sys::ModelingToolkit.NonlinearSystem`: Steady state system for `m`.

# Options
- `u0 = zeros(length(unknowns(sys)))`: Initial guess of steady state variables.
  `length(u0)` should be twice the number of variables in the model.
  The ordering of the elements of `u0` should be
  `[var1_level, var1_slope, var2_level, ..., varN_slope]`.
- `solver = nothing`: Solver to use.
  `nothing` means use the default solver determined by `solve`.
- Additional options are passed as keyword arguments to `solve`.
"""
function solve_steady_state! end
export solve_steady_state!

"""
    steady_state_system(m::Model)

Convert the steady state model associated with model `m`
into a `ModelingToolkit.NonlinearSystem`.

$MTK_NEEDED_STRING

# Example
This function is used to allow solving a `Model`'s steady state
using the ModelingToolkit/SciML ecosystem.
Here is an example of calling this function
and using [`solve_steady_state!`](@ref) on the result
to solve the model's steady state:
```julia
using ModelBaseEcon, StateSpaceEcon, ModelingToolkit, NonlinearSolve
@using_example E3
m = E3.newmodel()
sys = steady_state_system(m)
solver = NewtonRaphson() # Replace with desired solver.
solve_steady_state!(m, sys; solver)
```

---

    steady_state_system(sd::SteadyStateSolver.SolverData)

Alternative call signature when `sd` is already available (e.g., created manually).
"""
function steady_state_system end
export steady_state_system

"""
    compute_residuals_steady_state(u, sd::SteadyStateSolver.SolverData)

Compute the residuals of the steady state system in `sd`
given variable levels and slopes `u`.
This function is closed over in [`steady_state_system`](@ref)
to create a function that can be passed to `ModelingToolkit.NonlinearProblem`.

$MTK_NEEDED_STRING

!!! warning
    Internal function not part of the public interface.
"""
function compute_residuals_steady_state end

end

using .MTKExt

export stacked_time_system
export steady_state_system
export solve_steady_state!
