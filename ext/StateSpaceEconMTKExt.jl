module StateSpaceEconMTKExt

using ModelBaseEcon, StateSpaceEcon, ModelingToolkit
using ModelingToolkit.SciMLBase: solve, AbstractSolution
using StateSpaceEcon.StackedTimeSolver: StackedTimeSolverData
using StateSpaceEcon.SteadyStateSolver: SolverData as SteadyStateSolverData
using TimeSeriesEcon: rangeof, MVTSeries
import StateSpaceEcon.MTKExt:
    stacked_time_system,
    compute_residuals_stacked_time,
    _create_system,
    rename_variables,
    get_var_names,
    solve_steady_state!,
    steady_state_system,
    compute_residuals_steady_state,
    sol2data

# Note: Unfortunately, Documenter.jl seems not to include docstrings from package extensions.
# (At least it's not obvious how to do so.)
# As a result, the docstrings for the following methods live in `../src/MTKExt.jl`.

##############################
# Stacked time system
##############################

function stacked_time_system(m::Model, exog_data::AbstractArray{Float64,2}; fctype = fcgiven, plan::Plan=Plan(m, 1+m.maxlag:size(exog_data, 1)-m.maxlead))
    
    data = zeroarray(m, plan)
    data[plan.exogenous .== true] = exog_data[plan.exogenous .== true]
    data[1:m.maxlag, :] = exog_data[1:m.maxlag, :] # Initial conditions
    if fctype === fcgiven
        data[end-m.maxlead+1:end, :] = exog_data[end-m.maxlead+1:end, :] # Final conditions
    end

    # TODO: Should other variants be accepted, or is `:default` sufficient?
    sd = StackedTimeSolverData(m, plan, fctype, :default)

    return stacked_time_system(sd, data)

end

function stacked_time_system(sd::StackedTimeSolverData, data::AbstractArray{Float64,2})

    f = let sd = sd, data = data
        (u, p) -> compute_residuals_stacked_time(u, sd, data)
    end

    # Since we are just creating a MTK system out of the `NonlinearProblem`,
    # the `u0` we specify here is not actually used for solving the system.
    u0 = zeros(count(sd.solve_mask))
    prob = NonlinearProblem(f, u0)
    s = _create_system(prob, sd)
    s.defaults[:data_in] = data # hack to pass along the data to be used when constructing an MVTSeries

    return s

end

function compute_residuals_stacked_time(u, sd::StackedTimeSolverData, data::AbstractArray{Float64,2})

    # We need to use `zeros` instead of `similar` because `assign_exog_data!`
    # doesn't assign final conditions for the shocks.
    point = zeros(eltype(u), size(data))
    point[sd.solve_mask] = u
    # Note: `assign_exog_data!` assigns both exogenous data (including initial conditions) *and* final conditions.
    StateSpaceEcon.StackedTimeSolver.assign_exog_data!(point, data, sd)

    # Emulate `StackedTimeSolver.stackedtime_R!`, computing the residual
    # for each equation at each simulation period.
    resid = map(sd.TT[1:length(sd.II)]) do tt
        # `tt` grabs all the data from `point` for the current simulation period, lags, and leads (e.g., `tt = [1, 2, 3, 4, 5, 6]`).
        # The last value of `tt` contains the indices for just the leads (e.g., `tt = [100, 101, 102]`)
        # (but we don't use it in this loop).
        p = view(point, tt, :)
        med = sd.evaldata
        map(med.alleqns, med.allinds) do eqn, inds
            eqn.eval_resid(p[inds])
        end
    end

    return vcat(resid...)

end

function _create_system(prob::NonlinearProblem, sd::StackedTimeSolverData)

    # Convert `prob` into a ModelingToolkit system.
    @named old_sys = modelingtoolkitize(prob)

    # `modelingtoolkitize` creates a system with automatically generated variable names
    # `x₁`, `x₂`, etc.
    # Rename the variables to make it easier to query the solution to problems
    # created from this system.
    sys = rename_variables(old_sys, sd)

    # We need `conservative = true` to prevent errors in `structural_simplify`.
    s = structural_simplify(complete(sys); conservative = true)

    return s

end

function rename_variables(old_sys::NonlinearSystem, sd::StackedTimeSolverData)

    # reshape the solve_mask
    solve_mask = reshape(sd.solve_mask, :, maximum(last, sd.evaldata.var_to_idx))

    # Find all column indices of columns that have at least one `true`.
    var_cols = vec(mapslices(any, solve_mask; dims = 1)) |> findall
    
    # Convert from column indices to variable/shock names.
    var_names = map(var_cols) do i
        findfirst(p -> last(p) == i, sd.evaldata.var_to_idx)
    end

    var_lengths = Dict{Symbol,Int64}()
    col_count = 1
    for col in var_cols
        var_lengths[var_names[col_count]] = count(solve_mask[:, col])
        col_count += 1
    end
   
    # For each model variable, create a ModelingToolkit variable vector
    # with the correct number of time steps.
    # `vars` will end up being, e.g., `[pinf[1:97], rate[1:97], ygap[1:97]]`.
    vars = map(var_names) do var_name
        NE = var_lengths[var_name]
        ModelingToolkit.@variables $(var_name)[1:NE]
    end |> x -> reduce(vcat, x)
    
    # Collect all the individual variables to enable substituting the variable names
    # that were automatically generated by `modelingtoolkitize`.
    # `vars_enumerated` will end up being, e.g., `[pinf[1], pinf[2], ..., ygap[97]]`.
    vars_enumerated = reduce(vcat, map(collect, vars))
    old_vars = unknowns(old_sys)
    subs = Dict(old_vars .=> vars_enumerated)

    # Take all the equations from `old_sys`, rename the variables,
    # and create a new system with the new variable names.
    eqs = substitute.(ModelingToolkit.equations(old_sys), Ref(subs))
    @named sys = NonlinearSystem(eqs, vars_enumerated, [])

    return sys

end

function get_var_names(sd)

    # Reshape the solve mask to have a column for each variable/shock.
    solve_mask = reshape(sd.solve_mask, :, maximum(last, sd.evaldata.var_to_idx))
    # Find all column indices of columns that have at least one `true`.
    var_cols = vec(mapslices(any, solve_mask; dims = 1)) |> findall
    # Convert from column indices to variable/shock names.
    var_names = map(var_cols) do i
        findfirst(p -> last(p) == i, sd.evaldata.var_to_idx)
    end

    return var_names

end

##############################
# Steady state system
##############################

function solve_steady_state!(m::Model, sys::NonlinearSystem; u0 = zeros(length(unknowns(sys))), solver = nothing, solve_kwargs...)

    # Creating a `NonlinearProblem` from a `NonlinearFunction` seems to be faster
    # than creating a `NonlinearProblem` directly from a `NonlinearSystem`.
    nf = NonlinearFunction(sys)
    prob = NonlinearProblem(nf, u0)
    sol = isnothing(solver) ? solve(prob; solve_kwargs...) : solve(prob, solver; solve_kwargs...)

    # Copy the steady state solution to the model.
    m.sstate.values[.!m.sstate.mask] = sol.u
    # Mark the steady state as solved.
    m.sstate.mask .= true

    return m

end

steady_state_system(m::Model) = steady_state_system(SteadyStateSolverData(m))

function steady_state_system(sd::SteadyStateSolverData)

    f = let sd = sd
        (u, p) -> compute_residuals_steady_state(u, sd)
    end

    # Since we are just creating a MTK system out of the `NonlinearProblem`,
    # the `u0` we specify here is not actually used for solving the system.
    u0 = zeros(count(sd.solve_var))
    prob = NonlinearProblem(f, u0)
    @named sys = modelingtoolkitize(prob)
    s = complete(sys)

    return s

end

# `ModelBaseEcon.__to_dyn_pt` doesn't work with automatic differentiation,
# so copy it here to allow passing in an apropriately typed `buffer`.
# TODO: Should this method be added directly to ModelBaseEcon.jl?
function __to_dyn_pt!(buffer, pt, s)
    # This function applies the transformation from steady
    # state equation unknowns to dynamic equation unknowns
    for (i, jt) in enumerate(s.JT)
        if length(jt.ssinds) == 1
            pti = pt[jt.ssinds[1]]
        else
            pti = pt[jt.ssinds[1]] + ModelBaseEcon.__lag(jt, s) * pt[jt.ssinds[2]]
        end
        buffer[i] += pti
    end
    return buffer
end

function compute_residuals_steady_state(u, sd)

    # `point` needs to contain the levels and slopes for all variables and shocks,
    # but these values are all zero except for those included in `u`.
    point = zeros(eltype(u), length(sd.solve_var))
    point[sd.solve_var] = u

    # Emulate `SteadyStateSolver.global_SS_R!`, computing the residual for each equation.
    resid = map(sd.alleqns) do eqn
        if hasproperty(eqn.eval_resid, :s)
            # `eqn.eval_resid` closes over `s::SSEqData`.
            # There's no function to create just `s`, so just grab it from the closure.
            s = eqn.eval_resid.s
            buffer = zeros(eltype(u), length(s.JT))
            s.eqn.eval_resid(__to_dyn_pt!(buffer, point[eqn.vinds], s))
        else
            # `eqn.eval_resid` is an `EquationEvaluator` (not a closure)
            # that computes the residual for an equation defined by a steady state constraint.
            eqn.eval_resid(point[eqn.vinds])
        end
    end

    return resid

end

function sol2data(sol::AbstractSolution, s::ModelingToolkit.AbstractSystem, m::Model, p::Plan)
    rng = rangeof(p)
    
    # get endogenous points
    _result = ModelingToolkit.SymbolicIndexingInterface.getu(sol, 
        reduce(vcat, map(collect, map(var -> hasproperty(s, var.name) ? getproperty(s, var.name) : [], m.varshks)))
    )(sol) # is this a fair assumption, that we will get back the variables?
    
    # get mask for endogenous points
    mask = p.exogenous .== false
    mask[begin:begin+m.maxlag-1, :] .= false # initial conditions
    mask[end-m.maxlead+1:end, :] .= false # final conditions
    
    # construct results from exogenous data
    result = MVTSeries(rng, m.varshks, s.defaults[:data_in])

    # update endogenous points
    result[mask] = _result

    return result
end

end
