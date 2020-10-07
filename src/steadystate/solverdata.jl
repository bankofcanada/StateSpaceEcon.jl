
@inline ModelBaseEcon.geteqn(i::Integer, m::Model) = geteqn(i, m.sstate)

"""
    SolverData

A data structure used during the solution of the steady state problem.
It contains some current state information and some buffers.

!!! warning
    This is an internal data structure. Do not use directly.

### Why is this necessary?
The solution of the steady state problem is done in steps:
  1. The user sets the initial guess. This can be done with either `initial_sstate!` or `clear_sstate!`
  2. Pre-solve step. In this step we look for equations that have only one unknown and attempt to solve it.
  If successful, the unknown is marked as "solved" and is not an unknown anymore, also the equation is marked as
  "solved" and is not considered anymore. The process repeats for as long as it keeps solving.
  3. Solve step. This is where we solve the non-linear system composed of the remaining equations for the
  remaining variables.

This data structure keeps track of which unknowns and equations are solved and which remain to be solved.
It also holds an indexing map that translates the indexes of the variables and equations we solve for
to their original indexes in the full steady state system.

### Fields
  * `point` - a buffer for the current solution values. Some of these may be presolved, which are kept
  fixed while solving the system, while the rest are updated during solver iterations. The length equals
  the total number of steady state variables.
  * `resid` - a buffer for the current residual vector. The entries corresponding to presolved equations
  would normally be all zeros, while the ones corresponding to "active" equations would be updated during
  solver iterations. Lentgh equals the total number of steady state equations.
  * `solve_var` - a Boolean vector, same size as `point`. Value of `true` means that the unknown is "active", while `false` indicates
  that it has been pre-solved.
  * `solve_eqn` - a Boolean vector, same size as `resid`. Value of `true` means that the equation is "active", while `false` indicates
  that it has been pre-solved.
  * `vars_index` - an Integer vector, same length as `point`. Entries corresponding to pre-solved variables hold zeros.
  Entries for active vars are numbered sequentially from 1 to `nvars`
  * `eqns_index` - an Integer vector, same length as `resid`. Presolved equations have a zero here, while active equations
  are numbered sequentially from 1 to `neqns`.
"""
struct SolverData
    "Buffer holding the current solution."
    point::Vector{Float64}
    resid::Vector{Float64}
    "`true` for active vars and `false` for pre-solved vars."
    solve_var::Vector{Bool}
    "`true` for active equations and `false` for pre-solved equations."
    solve_eqn::Vector{Bool}
    "Sequential indexes of active vars. Pre-solved vars have index 0."
    vars_index::Vector{Int64}
    "Sequential indexes of active equations. Pre-solved equations have index 0."
    eqns_index::Vector{Int64}
    "The model equations to solve"
    alleqns::Vector{SteadyStateEquation}
end

function Base.getproperty(sd::SolverData, pname::Symbol)
    if pname == :nvars
        return sum(getfield(sd, :solve_var))
    elseif pname == :neqns
        return sum(getfield(sd, :solve_eqn))
    else
        return getfield(sd, pname)
    end
end

Base.propertynames(::SolverData) = (:nvars, :neqns, fieldnames(SolverData)...)

"""
    SolverData(model, presolve=Val(false); <options>)

Construct a SolverData instance from all variables and equations in the model,
ignoring anything pre-solved.

### Options
  `verbose::Bool` - if not specified it's taken from the model options.
  `tol::Float64` - desired tolerance when checking the residual of presolved equation.
  `presolve::Bool` - if `false`, any pre-solved information is ignored and
  the solver data is set up to solve all equations for all variables. 
"""
function SolverData(model::Model; presolve::Bool=true, 
            verbose::Bool=model.options.verbose,
            tol::Float64=model.options.tol
)
    # constructor where we're solving all equations for all variables
    local sstate = model.sstate
    local alleqns = ModelBaseEcon.alleqns(sstate)
    local neqns = length(alleqns)
    local nvars = length(sstate.values)
    sd = SolverData(copy(model.sstate.values), Vector{Float64}(undef, neqns),   # point and resid
                    Vector{Bool}(undef, nvars), Vector{Bool}(undef, neqns),     # solve_var and solve_eqn
                    Vector{Int64}(undef, nvars), Vector{Bool}(undef, neqns),    # vars_index and eqns_index
                    alleqns
    )
    if presolve
        # active vars are the ones with `sstate.mask` equal to `false`
        sd.solve_var .= .! sstate.mask
        # active equations are the ones that have any active vars
        for (i, eqn) in enumerate(sd.alleqns)
            sd.solve_eqn[i] = any(sd.solve_var[eqn.vinds])
        end
        # compute the residual at the initial point.
        global_SS_R!(sd.resid, sd.point, sd.alleqns)
        # collect the indexes of pre-solved equation with non-zero residuals.
        bad_eqn_inds = findall(@. (!sd.solve_eqn) & (abs(sd.resid) > tol))
        if !isempty(bad_eqn_inds)
            # Sort so that largest residuals are at the top
            sort!(bad_eqn_inds, lt=(l, r) -> abs(sd.resid[l]) > abs(sd.resid[r]))
            if verbose
                # print the list of bad equations
                sep = "\n    "
                bad_eqn_str = join(("E$i  res=$(sd.resid[i])  $(geteqn(i, sstate))" for i in bad_eqn_inds), sep)
                @warn "The following presolved equations are not satisfied.$(sep)$(bad_eqn_str)"
            end

            # Mark all bad equations and *all* their variables as active
            sd.solve_eqn[bad_eqn_inds] .= true
            for eqn in sd.alleqns[bad_eqn_inds]
                sd.solve_var[eqn.vinds] .= true
            end

            # Equations that are still pre-solved must have only pre-solved variables
            for eqn in sd.alleqns[.!sd.solve_eqn]
                sd.solve_var[eqn.vinds] .= false
            end

            # All masked variables (where sstate.mask == true) are still inactive as well
            sd.solve_var[sstate.mask] .= false

            # # if ssZeroSlope, make sure any slopes that were marked as active are restored.
            # if model.flags.ssZeroSlope
            #     sd.solve_var[2:2:end] .= false
            # end
        end
    else
        if model.flags.ssZeroSlope
            # ssZeroSlope means that all slopes are 0, so only the levels are active.
            sd.solve_var[1:2:end] .= true
            sd.solve_var[2:2:end] .= false
        else
            sd.solve_var .= true
        end
        # all shocks are zero and pre-solved
        for (i,v) in enumerate(model.allvars)
            if v.type == :shock 
                sd.solve_var[2i .+ (-1:0)] .= false
            elseif v.type == :steady
                sd.solve_eqn[2i] = false
            end
        end
        # all equations are active regardless.
        sd.solve_eqn .= true
        # compute the residual at the initial point.
        global_SS_R!(sd.resid, sd.point, sd.alleqns)
    end
    sd.vars_index .= 0
    sd.vars_index[sd.solve_var] .= 1:sum(sd.solve_var)
    sd.eqns_index .= 0
    sd.eqns_index[sd.solve_eqn] .= 1:sum(sd.solve_eqn)
    return sd
end
@assert precompile(SolverData, (Model,))


"""
    global_SS_RJ(point, sd::SolverData)

When applied to a solver data, computes the residual and Jacobian of the active set of equations
with respect to the active set of variables.
"""
function global_SS_RJ(point::AbstractVector{Float64}, sd::SolverData)
    sd.point[sd.solve_var] .= point
    R = zeros(sd.neqns)
    J = zeros(sd.neqns, sd.nvars)
    if sd.neqns == 0
        if sd.nvars > 0
            # Show an error message, but no exception is thrown
            @error "System is underdetermined"
        end
        return R, J
    end
    for (i, (solve, ind, eqn)) in enumerate(zip(sd.solve_eqn, sd.eqns_index, sd.alleqns))
        solve || continue
        rr, jj = try
            eqn.eval_RJ(sd.point[eqn.vinds]) 
        catch
            (NaN64, fill(NaN64, size(eqn.vinds)))
        end
        (isnan(rr) || isinf(rr)) && inadmissible_error(i, eqn, sd.point, rr)
        any(@. isnan(jj) | isinf(jj)) && inadmissible_error(i, eqn, sd.point, jj)
        R[ind] = rr
        # assign jj to the `ind`-th row in J.
        # jj contains the entire gradient. we need only the partials w.r.t. the active variables
        l_active = sd.solve_var[eqn.vinds]  # local mask the active variables
        l_index = sd.vars_index[eqn.vinds]  # local set of indexes (only active are valid, pre-solved are zero)
        J[ind, l_index[l_active]] .= jj[l_active]
    end
    return R, J
end
@assert precompile(global_SS_RJ, (Vector{Float64}, SolverData))

"""
    global_SS_RJ(point, sd::SolverData)

When a solver data is given, we compute the residual of the active equations only.
"""
function global_SS_R!(resid::AbstractVector{Float64}, point::AbstractVector{Float64},  sd::SolverData)
    sd.point[sd.solve_var] .= point
    for (i, (solve, ind, eqn)) in enumerate(zip(sd.solve_eqn, sd.eqns_index, sd.alleqns))
        solve || continue
        rr = try
            eqn.eval_resid(sd.point[eqn.vinds]) 
        catch
            NaN64
        end
        (isnan(rr) || isinf(rr)) && inadmissible_error(i, eqn, sd.point, rr)
        resid[ind] = rr
    end
    return nothing
end
@assert precompile(global_SS_R!, (Vector{Float64}, Vector{Float64}, SolverData))

