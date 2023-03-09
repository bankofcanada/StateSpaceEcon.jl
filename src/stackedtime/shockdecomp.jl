##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

import ..steadystatedata

"""
    shockdecomp(model, plan, exog_data; control, fctype, [options])

Compute the shock decomposition for the given model, plan, exogenous (shocks)
data and control solution.

If `control` option is not specified we use the steady state solution stored in
the model instance. The algorithm assumes that `control` is a solution to the
dynamic model for the given plan range and final condition. We verify the
residual and issue a warning, but do not enforce this. See
[`steadystatedata`](@ref).

As part of the algorithm we run a simulation with the given `plan`, 
`exog_data` and `fctype`.  See [`simulate`](@ref) for other options.

!!! note

    For now only the case of `anticipate=true` works. Shock decomp with
    unanticipated shocks is coming soon.

"""
function shockdecomp(m::Model, p::Plan, exog_data::SimData;
    initdecomp::Workspace,
    control::SimData,
    deviation::Bool=false,
    anticipate::Bool=true,
    variant::Symbol=m.options.variant,
    verbose::Bool=getoption(m, :verbose, false),
    maxiter::Int=getoption(m, :maxiter, 20),
    tol::Float64=getoption(m, :tol, 1e-12),
    sparse_solver::Function=(A, b) -> sparse_solve(A, b),
    linesearch::Bool=getoption(m, :linesearch, false),
    fctype::FinalCondition=getoption(m, :fctype, fcgiven),
    _debug=false
)

    refresh_med!(m, variant)

    if !anticipate
        error("Not implemented: `shockdecomp` with `solver=:stackedtime` and `anticipate=false`. If the model is linear, use `solver=:firstorder`.")
    end

    # if m.nauxs > 0
    #     error("Not yet implemented with auxiliary variables.")
    # end

    # we will return result at the end
    result = Workspace()
    result.c = copy(control)

    # apply data transformations of @log variables
    exog_data = transform(exog_data[p.range, :], m)
    control = transform(control[p.range, :], m)

    if deviation
        # with transformed data we simply add the control
        exog_data .+= control
    end

    # prepare the stacked-time
    gdata = StackedTimeSolverData(m, p, fctype, variant)

    # check the residual.           why are we doing this? we know it's 0!
    shocked = copy(exog_data)
    res_shocked = Vector{Float64}(undef, size(gdata.J, 1))
    stackedtime_R!(res_shocked, shocked, shocked, gdata)    # Run the "shocked" simulation with the given exogenous data.
    if norm(res_shocked, Inf) > tol
        assign_exog_data!(shocked, exog_data, gdata)
        sim_nr!(shocked, gdata, maxiter, tol, verbose, sparse_solver, linesearch)
    end

    # We need the Jacobian matrix evaluated at the control solution.
    res_control, _ = stackedtime_RJ(control, control, gdata)
    if norm(res_control, Inf) > tol
        # What to do if it's not a solution? We just give a warning for now, but maybe we should error()?!
        @warn "Control is not a solution:" norm(res_control, Inf)
    end

    # the difference between shocked and control solutions
    delta = shocked - control

    # now we need to split the unknowns into groups.
    begin
        LI = LinearIndices((p.range, 1:gdata.NU))
        NT = length(p.range)
        # range of initial conditions
        init = 1:m.maxlag
        # range of final conditions
        term = NT-m.maxlead+1:NT
        # simulation range
        sim = m.maxlag+1:NT-m.maxlead

        # indices of endogenous variable-points (the unknowns we are solving for)
        endo_inds = vec(vcat(
            (LI[sim, index] for (index, v) in enumerate(m.allvars) if !(isexog(v) || isshock(v)))...
        ))

        if length(endo_inds) != sum(gdata.solve_mask) || !all(gdata.solve_mask[endo_inds])
            error("Not yet implemented with non-trivial plan.")
        end

        # indices for the exogenous variable-points split into groups
        exog_inds = Workspace(;
            # contributions of initial conditions
            init=vec(LI[init, :]),
            # contributions of final conditions
            term=vec(LI[term, :]),
            # contributions of shocks
            (v.name => vec(LI[sim, index]) for (index, v) in enumerate(m.allvars) if isshock(v))...,
            # contributions of @exog variables
            (v.name => vec(LI[sim, index]) for (index, v) in enumerate(m.allvars) if isexog(v))...
        )
        # Note: we use a Workspace() because it preserves the order in which
        #     members were added to it

        if sum(length, values(exog_inds)) + length(endo_inds) != size(gdata.J, 2)
            error("endogenous and exogenous don't add up!")
        end
    end

    # now do the decomposition
    begin
        # we build a sparse matrix 
        SDI = Int[]     # row indexes
        SDJ = Int[]     # column indexes
        SDV = Float64[] # values

        # prep init (we build the result.sd on the fly)
        result.sd = Workspace()
        id = get(initdecomp, :sd, Workspace())
        pinit = p.range[init]
        if isempty(id)
            colind = 1
            inds = exog_inds[:init]
            append!(SDI, inds)
            append!(SDJ, fill(colind, size(inds)))
            append!(SDV, delta[inds])
            for v in m.allvars
                if isshock(v) || isexog(v)
                    continue
                end
                rsdv = result.sd[v.name] = MVTSeries(p.range, keys(exog_inds), zeros)
                rsdv[pinit, :init] = delta[pinit, v.name]
            end
        else
            shex = filter(v -> isshock(v) || isexog(v), m.allvars)
            for (vind, v) in enumerate(m.allvars)
                if isshock(v) || isexog(v)
                    continue
                end
                rsdv = result.sd[v.name] = MVTSeries(p.range, keys(exog_inds), zeros)
                if haskey(id, v.name)
                    # have initial decomposition
                    idv = id[v.name]
                    if norm(delta[pinit, v.name]-sum(idv[pinit, :],dims=2),Inf)>tol
                        error("The given `initdecomp` does not add up for $(v.name).")
                    end
                    rsdv[pinit] .= idv
                    for (colind, colname) in enumerate(keys(exog_inds))
                        if haskey(idv, colname) || colname ∈ shex
                            inds = LI[init, vind]
                            append!(SDI, inds)
                            append!(SDJ, fill(colind, size(inds)))
                            append!(SDV, idv[pinit, colname].values)
                        end
                    end
                else
                    # no initial decomposition - it all goes in init
                    rsdv[pinit, :init] = delta[pinit, v.name]
                    colind = 1
                    inds = LI[init, vind]
                    append!(SDI, inds)
                    append!(SDJ, fill(colind, size(inds)))
                    append!(SDV, delta[inds])
                end
            end
        end

        # prep shocks
        for (colind, (colname, inds)) in enumerate(pairs(exog_inds))
            if colname == :init
                continue
            end
            append!(SDI, inds)
            append!(SDJ, fill(colind, size(inds)))
            append!(SDV, delta[inds])
        end

        SDMAT = Matrix(gdata.J * sparse(SDI, SDJ, -SDV, size(gdata.J, 2), length(exog_inds)))
        factor = gdata.J_factorized[]
        if factor isa PardisoFactorization
            pardiso_solve!(factor, SDMAT)
        else
            ldiv!(factor, SDMAT)
        end

    end

    # now split and decorate the shock-decomposition matrix
    begin
        # in order to split the rows of SDMAT by variable, we need the inverse indexing map
        inv_endo_inds = zeros(Int, size(gdata.solve_mask))
        inv_endo_inds[gdata.solve_mask] .= 1:sum(gdata.solve_mask)
        for (index, v) in enumerate(m.allvars)
            v_inds = vec(LI[:, index])
            v_endo_mask = gdata.solve_mask[v_inds]
            if !any(v_endo_mask)
                continue
            end
            v_inv_endo_inds = inv_endo_inds[v_inds[v_endo_mask]]
            v_data = result.sd[v] 
            # assign contributions to endogenous points
            v_data[v_endo_mask, :] = SDMAT[v_inv_endo_inds, :]
            # contributions of initial conditions already assigned
            # assign contributions to final conditions
            if fctype === fcgiven || fctype === fclevel
                v_data[(begin-1).+term, :term] .= delta[v.name][term]
            elseif fctype === fcrate
                for tt in term
                    v_data[begin-1+tt, :] = v_data[begin-2+tt, :]
                end
            elseif fctype === fcnatural
                for tt in term
                    v_data[begin-1+tt, :] = 2 * v_data[begin-2+tt, :] - v_data[begin-3+tt, :]
                end
            else
                error("Not supported `fctype = $fctype` ")
            end
            # we also append the truncation error due to linear approximation of the decomposition.
            result.sd[v] = hcat(v_data; nonlinear=delta[v.name] - sum(v_data, dims=2))
        end
    end

    result.s = copy(result.c)
    result.s[p.range, :] = inverse_transform(shocked, m)

    if deviation
        logvars = islog.(m.varshks) .| isneglog.(m.varshks)
        result.s[:, logvars] ./= result.c[:, logvars]
        result.s[:, .!logvars] .-= result.c[:, .!logvars]
    end

    if _debug
        return result, gdata
    else
        return overlay(result, initdecomp)
    end
end


