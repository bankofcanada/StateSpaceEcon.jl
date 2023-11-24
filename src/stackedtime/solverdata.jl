##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

"""
    StackedTimeSolverData

The data structure used in the stacked time algorithm.
"""
struct StackedTimeSolverData # <: AbstractSolverData
    "Number of time periods"
    NT::Int
    "Number of variables"
    NV::Int
    "Number of shocks"
    NS::Int
    "Number of unknowns = variables + shocks + aux variables"
    NU::Int
    "Final condition type: `fcgiven` or `fclevel` or `fcslope`."
    FC::Vector{<:FinalCondition}
    "List of vectors for the time indexes of blocks of the matrix."
    TT::Vector{Vector{Int}}
    "List of vectors for the row indexes of each block."
    II::Vector{Vector{Int}}
    "List of vectors for the column indexes of each block."
    JJ::Vector{Vector{Int}}
    "Inverse lookup for the column indexes."
    BI::Vector{Vector{Int}}
    "Jacobian matrix cache."
    J::SparseMatrixCSC{Float64,Int}
    "Correction to J in case FC == fcslope. See description of algorithm."
    CiSc::SparseMatrixCSC{Float64,Int}   # Correction to J to account for FC in the case of fcslope
    "The steady state data, used for FC ∈ (fclevel, fcslope)."
    SS::AbstractArray{Float64,2}
    "Mask of variables with constant growth rate in the steady state"
    log_mask::AbstractVector{Bool}
    "Mask of variables with constant change in the steady state"
    lin_mask::AbstractVector{Bool}
    "The `evaldata` from the model, used to evaluate RJ and R!"
    evaldata::ModelBaseEcon.AbstractModelEvaluationData
    "Keep track of which variables are set by exogenous constraints."
    exog_mask::AbstractVector{Bool}
    "Keep track of which variables are set by final conditions."
    fc_mask::AbstractVector{Bool}
    "Keep track of which variables are \"active\" for the solver."
    solve_mask::AbstractVector{Bool}
    "Cache the factorization of the active part of J"
    J_factorized::Ref{Any}
    "Set to one of :qr or :lu"
    factorization::Symbol
end

#############################################################################

"""
    var_CiSc(sd::StackedTimeSolverData, var::ModelVariable, fc::FinalCondition)

Return data related to the correction of the Jacobian matrix needed for the
given final condition for the given variable.

!!! warning
    Internal function not part of the public interface.
"""
function var_CiSc end

"""
    assign_fc!(x::Vector, exog::Vector, vind::Int, sd::StackedTimeSolverData, fc::FinalCondition)

Applying the final condition `fc` for variable with index `vind`. Exogenous data
is provided in `exog` and stacked time solver data in `sd`. This function
updates the solution vector `x` in place and returns `x`.

!!! warning
    Internal function not part of the public interface.
"""
function assign_fc! end

# var_CiSc(::StackedTimeSolverData, ::ModelVariable, ::FinalCondition) = error("Missing var_CiSc for $(typeof(fc))")
# assign_fc!(::AbstractVector{Float64}, ::AbstractVector{Float64}, ::Int, fc::FinalCondition) = error("Missing assign_fc! for $(typeof(fc))")

#######

@inline function var_CiSc(::StackedTimeSolverData, ::ModelVariable, ::FCNone)
    # No Jacobian correction
    return Dict{Int,Vector{Float64}}()
end

@inline function assign_fc!(x::AbstractVector{Float64}, exog::AbstractVector{Float64}, vind::Int, sd::StackedTimeSolverData, ::FCNone)
    # nothing to see here
    return x
end

#######

@inline function var_CiSc(::StackedTimeSolverData, ::ModelVariable, ::FCGiven)
    # No Jacobian correction
    return Dict{Int,Vector{Float64}}()
end

@inline function assign_fc!(x::AbstractVector{Float64}, exog::AbstractVector{Float64}, ::Int, sd::StackedTimeSolverData, ::FCGiven)
    # values given exogenously
    if x !== exog
        x[sd.TT[end]] .= exog[sd.TT[end]]
    end
    return x
end

#######

@inline function var_CiSc(::StackedTimeSolverData, ::ModelVariable, ::FCMatchSSLevel)
    # No Jacobian correction
    return Dict{Int,Vector{Float64}}()
end

@inline function assign_fc!(x::AbstractVector{Float64}, ::AbstractVector{Float64}, vind::Int, sd::StackedTimeSolverData, ::FCMatchSSLevel)
    # values come from the steady state
    x[sd.TT[end]] .= sd.SS[sd.TT[end], vind]
    return x
end

#######

@inline function var_CiSc(sd::StackedTimeSolverData, ::ModelVariable, ::FCMatchSSRate)
    # Matrix C is composed of blocks looking like this:
    #    1  0  0  0
    #   -1  1  0  0
    #    0 -1  1  0
    #    0  0 -1  1
    # The corresponding rows in Sc look like this
    #   0 0 0 0 0 -1
    #   0 0 0 0 0  0
    #   0 0 0 0 0  0
    #   0 0 0 0 0  0
    # The C^{-1} * Sc matrix then looks like this
    #   0 0 0 0 0 -1
    #   0 0 0 0 0 -1
    #   0 0 0 0 0 -1
    #   0 0 0 0 0 -1
    # The number of rows equals NTFC (the number of final condition periods)
    # The column with the -1 is the one corresponding to the last simulation period of the variable.
    return Dict{Int,Vector{Float64}}(0 => fill(-1.0, length(sd.TT[end])))
end

@inline function assign_fc!(x::AbstractVector{Float64}, ::AbstractVector{Float64}, vind::Int, sd::StackedTimeSolverData, ::FCMatchSSRate)
    # compute values using the slope from the steady state
    for t in sd.TT[end]
        x[t] = x[t-1] + sd.SS[t, vind] - sd.SS[t-1, vind]
    end
    return x
end

#######

@inline function var_CiSc(sd::StackedTimeSolverData, ::ModelVariable, ::FCConstRate)
    # Matrix C is composed of blocks looking like this:
    #    -1  0  0  0  0
    #     2 -1  1  0  0
    #    -1  2 -1  0  0
    #     0 -1  2 -1  0
    #     0  0 -1  2 -1
    # The corresponding rows in Sc look like this
    #     0 ..  0 -1  2
    #     0 ..  0  0 -1
    #     0 ..  0  0  0
    #     0 ..  0  0  0
    #     0 ..  0  0  0
    # The C^{-1} * Sc matrix then looks like this
    #     0 ..  0  1 -2
    #     0 ..  0  2 -3
    #     0 ..  0  3 -4
    #     0 ..  0  4 -5
    #     0 ..  0  5 -6

    # The number of rows equals sd.TT[end] (the number of final condition periods)
    # The column with -2,-3,-4... has offset 0 -- the last simulation period.
    # The column with  1, 2, 3... has offset -1 -- the period before the last simulation period.
    foo = collect(Float64, 1:length(sd.TT[end]))
    return Dict{Int,Vector{Float64}}(-1 => foo, 0 => -1.0 .- foo)
end

@inline function assign_fc!(x::AbstractVector{Float64}, ::AbstractVector{Float64}, ::Int, sd::StackedTimeSolverData, ::FCConstRate)
    # compute values using the slope at the end of the simulation
    for t in sd.TT[end]
        x[t] = 2x[t-1] - x[t-2]
    end
    return x
end


"""
    update_plan!(sd::StackedTimeSolverData, model, plan; changed=false)

Update the stacked time solver data to reflect the new plan. The new plan must
have the same range as the original plan, otherwise the solver data cannot be
updated in place.

By default the data structure is updated only if an actual change in the plan is
detected. Setting the `changed` flag to `true` forces the update even if the
plan seems unchanged. This is necessary only in rare circumstances.

!!! warning
    Internal function not part of the public interface.
"""
function update_plan!(sd::StackedTimeSolverData, model::Model, plan::Plan; changed=false)
    if sd.NT != length(plan.range)
        error("Unable to update using a simulation plan of different length.")
    end

    unknowns = model.allvars

    # LinearIndices used for indexing the columns of the global matrix
    LI = LinearIndices((plan.range, 1:sd.NU))

    sim = model.maxlag+1:sd.NT-model.maxlead
    NTFC = model.maxlead

    # Assume initial conditions are set correctly to exogenous in the constructor
    # We only update the masks during the simulation periods
    foo = BitArray(undef, sd.NU)
    for t in sim
        # s = p[t]
        # si = indexin(s, unknowns)
        si = plan[t, Val(:inds)]
        fill!(foo, false)
        foo[si] .= true
        if !all(foo .== sd.exog_mask[LI[t, :]])
            sd.exog_mask[LI[t, :]] .= foo
            changed = true
        end
    end

    # Update the solve_mask array
    if changed
        @. sd.solve_mask = !(sd.fc_mask | sd.exog_mask)
        @assert !any(sd.exog_mask .& sd.fc_mask)
        @assert sum(sd.solve_mask) == size(sd.J, 1)
        sm_index = cumsum(sd.solve_mask)

        # Update the Jacobian correction matrix, if exogenous plan changed
        II, JJ, VV = Int[], Int[], Float64[]
        last_sim_t = last(sim)
        var_to_idx = ModelBaseEcon.get_var_to_idx(model)
        JJvar = fill(0, NTFC)
        for (vi, (v, fc)) in enumerate(zip(unknowns, sd.FC))
            @assert vi == var_to_idx[v]
            # var_CiSc returns a Dict with keys equal to the column offset relative to last_sim_t
            # and values containing the column values.
            for (offset, values) in var_CiSc(sd, v, fc)
                # col_ind is the column index in the global matrix
                col_ind = LI[last_sim_t+offset, vi]
                if !sd.solve_mask[col_ind]
                    # this value is exogenous, no column for it in Sc
                    continue
                end
                # The column index in the active matrix
                # JJvar = begin
                #     # Full matrix has sd.NU*sd.NT columns
                #     # We take a vector with this many `false`s and put a `true` only in `col_ind` position
                #     # Then we select only the active positions in this vector (that's foo[sd.solve_mask])
                #     # Then we find the index of the `true` value - that's the column index in the active sub-matrix.
                #     foo = falses(sd.NU * sd.NT)
                #     foo[col_ind] = true
                #     ind = findall(foo[sd.solve_mask])
                #     repeat(ind, NTFC)
                # end
                fill!(JJvar, sm_index[col_ind])
                # The row indices for this block
                #   the block for variable 1 gets row indexes 1 .. NTFC
                #   the block for variable 2 gets row indexes NTFC+1 .. 2*NTFC
                #   the block for variable vi gets row indexes (vi-1)*NTFC+1 .. vi*NTFC
                IIvar = ((vi-1)*NTFC+1):(vi*NTFC)
                # The values in this column came from the var_CiSc call above
                VVvar = values
                append!(II, IIvar)
                append!(JJ, JJvar)
                append!(VV, VVvar)
            end
        end
        # Construct the sparse matrix
        sd.CiSc .= sparse(II, JJ, VV, NTFC * sd.NU, size(sd.J, 1))
        # cached lu is no longer valid, since active columns have changed
        sd.J_factorized[] = nothing
    end

    return sd
end

"""
    make_BI(J, II)

Prepares the `BI` array for the solver data. Called from the constructor of
`StackedTimeSolverData`.

!!! warning
    Internal function not part of the public interface.
"""
function make_BI(JMAT::SparseMatrixCSC{Float64,Int}, II::AbstractVector{<:AbstractVector{Int}})
    # Computes the set of indexes in JMAT.nzval corresponding to blocks of equations in II

    # Bar is the inverse map of II, i.e. Bar[j] = i <=> II[i] contains j
    local Bar = zeros(Int, size(JMAT, 1))
    for i in axes(II, 1)
        Bar[II[i]] .= i
    end

    # start with empty lists in BI
    local BI = [[] for _ in II]

    # iterate the non-zero elements of JMAT and record their row numbers in BI
    local rows = rowvals(JMAT)
    local n = size(JMAT, 2)
    for i = 1:n
        for j in nzrange(JMAT, i)
            row = rows[j]
            push!(BI[Bar[row]], j)
        end
    end
    return BI
end

StackedTimeSolverData(model::Model, plan::Plan, fctype::FinalCondition, variant::Symbol=model.options.variant) = StackedTimeSolverData(model, plan, setfc(model, fctype), variant)
@timeit_debug timer function StackedTimeSolverData(model::Model, plan::Plan, fctype::AbstractVector{FinalCondition}, variant::Symbol=model.options.variant)

    evaldata = getevaldata(model, variant)
    var_to_idx = ModelBaseEcon.get_var_to_idx(model)

    NT = length(plan.range)
    init = 1:model.maxlag
    term = NT-model.maxlead+1:NT
    sim = model.maxlag+1:NT-model.maxlead
    NTFC = length(term)
    NTSIM = length(sim)
    NTIC = length(init)

    need_SS = false
    for fc in fctype
        if (fc === fclevel || fc === fcslope)
            need_SS = true
        end
        if (fc === fcnatural) && (length(sim) + length(init) < 2)
            throw(ArgumentError("Simulation must include at least 2 periods for `fcnatural`."))
        end
    end

    if need_SS && !issssolved(model)
        # NOTE: we do not verify the steady state, just make sure it's been assigned
        throw(ArgumentError("Steady state must be solved for `fclevel` or `fcslope`."))
    end

    unknowns = model.allvars
    nunknowns = length(unknowns)

    nvars = length(model.variables)
    nshks = length(model.shocks)
    nauxs = length(model.auxvars)

    equations = model.alleqns
    nequations = length(equations)

    # LinearIndices used for indexing the columns of the global matrix
    LI = LinearIndices((plan.range, 1:nunknowns))

    # Initialize empty arrays
    TT = Vector{Int}[] # the time indexes of the block, used when updating values in stackedtime_RJ
    II = Vector{Int}[] # the row indexes
    JJ = Vector{Int}[] # the column indexes

    # Prep the Jacobian matrix
    neq = 0 # running counter of equations added to matrix
    # Model equations are the same for each sim period, just shifted according to t
    Jblock = [ti + NT * (var_to_idx[var] - 1) for (_, eqn) in equations for (var, ti) in keys(eqn.tsrefs)]
    Iblock = [i for (i, (_, eqn)) in enumerate(equations) for _ in eqn.tsrefs]

    Tblock = -model.maxlag:model.maxlead
    for t in sim
        push!(TT, t .+ Tblock)
        push!(II, neq .+ Iblock)
        push!(JJ, t .+ Jblock)
        neq += nequations
    end
    # Construct the
    begin
        I = vcat(II...)
        J = vcat(JJ...)
        JMAT = sparse(I, J, fill(NaN64, size(I)), NTSIM * nequations, NT * nunknowns)
    end

    # We no longer need the exact indexes of all non-zero entires in the Jacobian matrix.
    # We do however need the set of equation indexes for each sim period
    foreach(unique! ∘ sort!, II)

    # BI holds the indexes in JMAT.nzval for each block of equations
    BI = make_BI(JMAT, II)  # same as the two lines above, but faster

    # We also need the times of the final conditions
    push!(TT, term)

    #
    exog_mask = falses(nunknowns * NT)
    # Initial conditions are set as exogenous values
    exog_mask[vec(LI[init, :])] .= true
    # The exogenous values during sim are set in update_plan!() call below.

    # Final conditions are complicated
    fc_mask = falses(nunknowns * NT)
    fc_mask[vec(LI[term, :])] .= true

    # The solve_mask is redundant. We pre-compute and store it for speed
    solve_mask = .!(fc_mask .| exog_mask)

    sd = StackedTimeSolverData(NT, nvars, nshks, nunknowns, fctype, TT, II, JJ, BI, JMAT,
        sparse([], [], Float64[], NTFC * nunknowns, size(JMAT, 1)),
        need_SS ? transform(steadystatearray(model, plan), model) : zeros(0, 0),
        islog.(model.allvars) .| isneglog.(model.allvars), islin.(model.allvars),
        evaldata, exog_mask, fc_mask, solve_mask,
        Ref{Any}(nothing), getoption(model, :factorization, :default))

    return update_plan!(sd, model, plan; changed=true)
end

"""
    assign_exog_data!(x::Matrix, exog::Matrix, sd::StackedTimeSolverData)

Assign the exogenous points into `x` according to the plan with which `sd` was created using
exogenous data from `exog`.  Also call [`assign_final_condition!`](@ref).

!!! warning
    Internal function not part of the public interface.
"""
function assign_exog_data!(x::AbstractArray{Float64,2}, exog::AbstractArray{Float64,2}, sd::StackedTimeSolverData)
    # @assert size(x,1) == size(exog,1) == sd.NT
    x[sd.exog_mask] = exog[sd.exog_mask]
    assign_final_condition!(x, exog, sd)
    return x
end

"""
    assign_final_condition!(x::Matrix, exog::Matrix, sd::StackedTimeSolver)

Assign the final conditions into `x`. The final condition types for the different variables of the model
are stored in the the solver data `sd`. `exog` is used for [`fcgiven`](@ref).

!!! warning
    Internal function not part of the public interface.
"""
function assign_final_condition!(x::AbstractArray{Float64,2}, exog::AbstractArray{Float64,2}, sd::StackedTimeSolverData)
    for (vi, fc) in enumerate(sd.FC)
        assign_fc!(view(x, :, vi), exog[:, vi], vi, sd, fc)
    end
    return x
end

"""
    stackedtime_R!(R::Vector, point::Array, exog::Array, sd::StackedTimeSolverData)

Compute the residual of the stacked time system at the given `point`. R is
updated in place and returned.
"""
@timeit_debug timer function stackedtime_R!(res::AbstractArray{Float64,1}, point::AbstractArray{Float64}, exog_data::AbstractArray{Float64}, sd::StackedTimeSolverData)
    @assert size(point) == size(exog_data) == (sd.NT, sd.NU)
    # point = reshape(point, sd.NT, sd.NU)
    # exog_data = reshape(exog_data, sd.NT, sd.NU)
    @assert(length(res) == size(sd.J, 1), "Length of residual vector doesn't match.")
    for (ii, tt) in zip(sd.II, sd.TT)
        eval_R!(view(res, ii), point[tt, :], sd.evaldata)
    end
    return res
end

#= disable

# this is not necessary with log-transformed variables
# we keep it because it might be necessary for other transformations in the future.

@inline update_CiSc!(x::AbstractArray{Float64,2}, sd::StackedTimeSolverData) = any(sd.log_mask) ? update_CiSc!(x, sd, Val(sd.FC)) : nothing

@inline update_CiSc!(x, sd, ::Val{fcgiven}) = nothing
@inline update_CiSc!(x, sd, ::Val{fclevel}) = nothing
function update_CiSc!(x, sd::StackedTimeSolverData, ::Val{fcslope})
    # LinearIndices used for indexing the columns of the global matrix
    local LI = LinearIndices((1:sd.NT, 1:sd.NU))
    # The last simulation period
    local term_Ts = sd.TT[end]
    if isempty(term_Ts)
        return
    end
    local last_T = term_Ts[1] - 1
    local NTFC = length(term_Ts)

    # @info "last_T = $(last_T), NTFC = $(NTFC)"

    # Iterate over the non-zero values of CiSc and update
    # values as necessary
    local rows = rowvals(sd.CiSc)
    local vals = nonzeros(sd.CiSc)
    local ncols = size(sd.CiSc, 2)
    for col = 1:ncols
        nzr = nzrange(sd.CiSc, col)
        if isempty(nzr)
            continue
        end
        # Compute variable and time index from column index
        var, tm = divrem(LI[sd.solve_mask][col] - 1, sd.NT)  .+ 1

        # @info "var = $(var), tm = $(tm)"

        if !sd.log_mask[var]
            continue
        end
        if tm != last_T
            error("Last simulation times don't match")
        end
        for j = nzr
            row = rows[j]
            # update vals[j]
            local T = rem(row - 1, NTFC) + 1 + last_T

            # @info "Updating $((row, col)) with T=$(T), x[T, var] = $(x[T, var]), x[last_T, var] = $(x[last_T, var])"

            vals[j] = - x[T, var] / x[last_T, var]
        end
    end
    return nothing
end

function update_CiSc!(x, sd, ::Val{fcnatural})
    # LinearIndices used for indexing the columns of the global matrix
    local LI = LinearIndices((1:sd.NT, 1:sd.NU))
    # The last simulation period
    local term_Ts = sd.TT[end]
    if isempty(term_Ts)
        return
    end
    local last_T = term_Ts[1] - 1
    local NTFC = length(term_Ts)

    # @info "last_T = $(last_T), NTFC = $(NTFC)"

    # Iterate over the non-zero values of CiSc and update
    # values as necessary
    local rows = rowvals(sd.CiSc)
    local vals = nonzeros(sd.CiSc)
    local ncols = size(sd.CiSc, 2)
    for col = 1:ncols
        nzr = nzrange(sd.CiSc, col)
        if isempty(nzr)
            continue
        end
        # Compute variable and time index from column index
        var, tm = divrem(LI[sd.solve_mask][col] - 1, sd.NT)  .+ 1

        # @info "var = $(var), tm = $(tm)"

        if !sd.log_mask[var]
            continue
        end
        if tm == last_T
            s = -1
            c = 2
        elseif tm == last_T - 1
            s = +1
            c = 1
        else
            error("Last simulation times don't match")
        end
        for j = nzr
            row = rows[j]
            # update vals[j]
            local T = rem(row - 1, NTFC) + 1 + last_T

            # @info "Updating $((row, col)) with T=$(T), x[T, var] = $(x[T, var]), x[last_T, var] = $(x[last_T, var])"

            vals[j] = s * (c + T - last_T - 1 ) * x[T, var] / x[tm, var]
        end
    end
    return nothing
end
=#


"""
    R, J = stackedtime_RJ(point::Array, exog::Array, sd::StackedTimeSolverData)

Compute the residual and Jacobian of the stacked time system at the given
`point`.
"""
@timeit_debug timer function stackedtime_RJ(point::AbstractArray{Float64}, exog_data::AbstractArray{Float64}, sd::StackedTimeSolverData;
    debugging=false, factorization=sd.factorization)
    nunknowns = sd.NU
    @assert size(point) == size(exog_data) == (sd.NT, nunknowns)
    # point = reshape(point, sd.NT, nunknowns)
    # exog_data = reshape(exog_data, sd.NT, nunknowns)
    JAC = sd.J
    RES = Vector{Float64}(undef, size(JAC, 1))
    # Model equations @ [1] to [end-1]
    haveJ = isa(sd.evaldata, ModelBaseEcon.LinearizedModelEvaluationData) && !any(isnan.(JAC.nzval))
    haveLU = sd.J_factorized[] !== nothing
    if haveJ
        # update only RES
        for i = 1:length(sd.BI)
            eval_R!(view(RES, sd.II[i]), point[sd.TT[i], :], sd.evaldata)
        end
    else
        # update both RES and JAC
        for i = 1:length(sd.BI)
            R, J = eval_RJ(point[sd.TT[i], :], sd.evaldata)
            RES[sd.II[i]] .= R
            JAC.nzval[sd.BI[i]] .= J.nzval
        end
        # update_CiSc!(point, sd)
        haveLU = false
    end
    if !haveLU
        if nnz(sd.CiSc) > 0
            JJ = JAC[:, sd.solve_mask] - JAC[:, sd.fc_mask] * sd.CiSc
        else
            JJ = JAC[:, sd.solve_mask]
        end
        # compute factorization of the active part of J and cache it.
        sf_factorize!(Val(factorization), sd.J_factorized, JJ)
    end
    return RES, sd.J_factorized[]
end

function sf_factorize!(v::Val, Rf::Ref{Any}, A::SparseMatrixCSC)
    if isnothing(Rf[])
        Rf[] = sf_prepare(v, A)
    else
        Rf[] = sf_factor!(Rf[], A)
    end
end


#= KristofferC' implemetation (delete eventually, but keep it for now, just in case)

using Pardiso

mutable struct PardisoFactorization
    ps::MKLPardisoSolver
    J::SparseMatrixCSC
    PardisoFactorization(ps::MKLPardisoSolver) = new(ps)
end

@timeit_debug timer  function pardiso_init()
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_NONSYM)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    # See https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html
    set_iparm!(ps, 2, 2) # The parallel (OpenMP) version of the nested dissection algorithm.
    psf = PardisoFactorization(ps)
    finalizer(psf) do x
        set_phase!(x.ps, Pardiso.RELEASE_ALL)
        pardiso(x.ps)
    end
    return psf
end


# See https://github.com/JuliaSparse/Pardiso.jl/blob/master/examples/exampleunsym.jl
@timeit_debug timer  function pardiso_factorize(JJ; psf::Union{Nothing,PardisoFactorization})
    reuse_ps = psf !== nothing
    if psf === nothing
        psf = pardiso_init()
    end

    ps = psf.ps
    psf.J = get_matrix(ps, JJ, :N)

    # No need to run the analysis phase if the sparsity pattern is unchanged
    # and we reuse the same solver object
    if !reuse_ps
        set_phase!(ps, Pardiso.ANALYSIS)
        pardiso(ps, psf.J, Float64[])
    end
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, psf.J, Float64[])
    return psf
end

@timeit_debug timer  function pardiso_solve!(ps::PardisoFactorization, x::AbstractArray)
    ps, J = ps.ps, ps.J
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    X = similar(x) # Solution is stored in X
    pardiso(ps, X, J, x)
    copy!(x, X)
end


@timeit_debug timer  function _factorize(JJ; psf)
     if USE_PARDISO_FOR_LU
        return pardiso_factorize(JJ; psf)
    else
        return lu(JJ)
    end
end

=#

"""
    assign_update_step!(x::Array, lambda, dx, sd::StackedTimeSolverData)

Perform something similar to `x = x + lambda * dx`, but with the necessary
corrections related to final conditions.
"""
function assign_update_step!(x::AbstractArray{Float64}, λ::Float64, Δx::AbstractArray{Float64}, sd::StackedTimeSolverData)
    x[sd.solve_mask] .+= λ .* Δx
    if nnz(sd.CiSc) > 0
        x[sd.fc_mask] .-= λ .* (sd.CiSc * Δx)
        # if sd.FC == fcnatural && any(sd.log_mask) && !isempty(sd.TT[end])
        #     for t = sd.TT[end]
        #         @. x[t, sd.log_mask] = x[t - 1, sd.log_mask]^2 / x[t - 2,sd.log_mask]
        #     end
        # end
        # assign_final_condition!(x, zeros(0,0), sd, Val(fcslope))
    end
    return x
end
