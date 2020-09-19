
# export StackedTimeSolverData
"""
    StackedTimeSolverData <: AbstractSolverData

The data structure used in the stacked time algorithm.

**TODO** Add all the details here.
"""
struct StackedTimeSolverData <: AbstractSolverData
    "Number of time periods"
    NT::Int64
    "Number of variables"
    NV::Int64
    "Number of shocks"
    NS::Int64
    "Number of unknowns = variables + shocks + aux variables"
    NU::Int64
    "Final condition type: `fcgiven` or `fclevel` or `fcslope`."
    FC::FCType
    "List of vectors for the time indexes of blocks of the matrix."
    TT::Vector{Vector{Int64}}
    "List of vectors for the row indexes of each block."
    II::Vector{Vector{Int64}}
    "List of vectors for the column indexes of each block."
    JJ::Vector{Vector{Int64}}
    "Inverse lookup for the column indexes."
    BI::Vector{Vector{Int64}}
    "Jacobian matrix cache."
    J::SparseMatrixCSC{Float64,Int64}
    "Correction to J in case FC == fcslope. See description of algorithm."
    CiSc::SparseMatrixCSC{Float64,Int64}   # Correction to J to account for FC in the case of fcslope
    "The steady state data, used for FC ∈ (fclevel, fcslope)."
    SSV::AbstractVector{Float64}
    "The `evaldata` from the model, used to evaluate RJ and R!"
    evaldata::ModelBaseEcon.AbstractModelEvaluationData
    "Keep track of which variables are set by exogenous constraints."
    exog_mask::Vector{Bool}
    "Keep track of which variables are set by final conditions."
    fc_mask::Vector{Bool}
    "Keep track of which variables are \"active\" for the solver."
    solve_mask::Vector{Bool}
    "Cache the LU decomposition of the active part of J"
    luJ::Ref{Any}
end


"""
    update_plan!(sd::StackedTimeSolverData, model, plan; changed=false)

Implementation for the Stacked Time algorithm. Plan must have the same range as
the original plan.

By default the data structure is updated only if an actual change in the plan is
detected. Setting the `changed` flag to `true` forces the update even if the
plan seems unchanged. This is necessary only in rare circumstances.

"""
function update_plan!(sd::StackedTimeSolverData, m::Model, p::Plan; changed=false)
    if sd.NT != length(p.range)
        error("Unable to update using a simulation plan of different length.")
    end

    # LinearIndices used for indexing the columns of the global matrix
    LI = LinearIndices((p.range, 1:sd.NU))

    sim = m.maxlag + 1:sd.NT - m.maxlead
    NTFC = m.maxlead

    # Assume initial conditions are set correctly to exogenous in the constructor
    # We only update the masks during the simulation periods
    # unknowns = m.allvars
    foo = falses(sd.NU)
    for t in sim
        # s = p[t]
        # si = indexin(s, unknowns)
        si = p[t, Val(:inds)]
        fill!(foo, false)
        foo[si] .= true
        if !all(foo .== sd.exog_mask[LI[t,:]])
            sd.exog_mask[LI[t,:]] .= foo
            changed = true
        end
    end

    # Update the solve_mask array
    if changed
        @. sd.solve_mask = !(sd.fc_mask | sd.exog_mask)
        @assert !any(sd.exog_mask .& sd.fc_mask)
        @assert sum(sd.solve_mask) == size(sd.J, 1)
    end

    # Update the Jacobian correction matrix, if exogenous plan changed
    if changed && (sd.FC == fcslope)
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

        # # foo is a mask array with foo[i] = true if i is a column  index for the last simulation period of a variable and foo[i] = false otherwise.
        # foo = falses(sd.NU * sd.NT)                 # all false
        # foo[LI[last(sim),1:sd.NV]] .= true          # true at last sim period of variables
        # foo[LI[last(sim),sd.NV + sd.NS + 1:sd.NU]] .= true  # true at last sim period of aux variables
        # # no final conditions for shocks - their final condition is always 0 (the expectation of the shock).

        # # The column indices - findall(foo) returns the indexes of the elements that are true.
        # #   then we repeat each NTFC times, since this is the number of rows we need
        # JJfoo = repeat(findall(foo[sd.solve_mask]), inner=NTFC)

        # The row indices
        #   the block for variable 1 gets row indexes 1 .. NTFC
        #   the block for variable 2 gets row indexes NTFC+1 .. 2*NTFC
        #   the block for variable vi gets row indexes (vi-1)*NTFC+1 .. vi*NTFC
        # IIfoo = vcat((NTFC * (vi - 1) .+ (1:NTFC) for vi in Iterators.flatten((1:sd.NV, sd.NV + sd.NS + 1:sd.NU)) if sd.solve_mask[LI[last(sim),vi]])...)
        IIfoo = Int[]
        JJfoo = Int[]
        VVfoo = Float64[]
        for vi in Iterators.flatten((1:sd.NV, (sd.NV + sd.NS + 1):sd.NU))
            colind = LI[last(sim), vi]
            if sd.solve_mask[colind]
                # if the variable is exogenous in the last period, then Sc doesn't have a column for it.
                append!(IIfoo, ((vi - 1) * NTFC + 1):(vi * NTFC))
                append!(VVfoo, fill(-1.0, NTFC))
                foo = falses(sd.NU * sd.NT)
                foo[colind] = true
                append!(JJfoo, repeat(findall(foo[sd.solve_mask]), NTFC))
            end
        end
        # Construct the sparse matrix
        sd.CiSc .= sparse(IIfoo, JJfoo, VVfoo, NTFC * sd.NU, size(sd.J, 1))
        # sd.CiSc .= sparse(IIfoo, JJfoo, fill(Float64(-1), length(JJfoo)), NTFC * sd.NU, size(sd.J, 1))
        # sd.CiSc.nzval .= -1
    end
        # Update the Jacobian correction matrix, if exogenous plan changed
    if changed && (sd.FC == fcnatural)
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

        # The number of rows equals NTFC (the number of final condition periods)
        # The column with the -2,-3,-4... is the one corresponding to the last simulation period of the variable.
        # The column with the 1,2,3... is the one corresponding to the period before the last simulation period of the variable.

        # The row indices
        #   the block for variable 1 gets row indexes 1 .. NTFC
        #   the block for variable 2 gets row indexes NTFC+1 .. 2*NTFC
        #   the block for variable vi gets row indexes (vi-1)*NTFC+1 .. vi*NTFC
        # IIfoo = vcat((NTFC * (vi - 1) .+ (1:NTFC) for vi in Iterators.flatten((1:sd.NV, sd.NV + sd.NS + 1:sd.NU)) if sd.solve_mask[LI[last(sim),vi]])...)
        IIfoo = Int[]
        JJfoo = Int[]
        VVfoo = Float64[]
        for vi in Iterators.flatten((1:sd.NV, (sd.NV + sd.NS + 1):sd.NU))
            let col1 = LI[last(sim) - 1, vi]
                if sd.solve_mask[col1]
                    append!(IIfoo, ((vi - 1) * NTFC + 1):(vi * NTFC))
                    append!(VVfoo, 1:NTFC)
                    foo = falses(sd.NU * sd.NT)
                    foo[col1] = true
                    append!(JJfoo, repeat(findall(foo[sd.solve_mask]), NTFC))
                end
            end
            let col2 = LI[last(sim), vi]
                if sd.solve_mask[col2]
                # if the variable is exogenous in the last period, then Sc doesn't have a column for it.
                    append!(IIfoo, ((vi - 1) * NTFC + 1):(vi * NTFC))
                    append!(VVfoo, -1.0 .- (1:NTFC))   # -2, -3, ..., -NTFC-1
                    foo = falses(sd.NU * sd.NT)
                    foo[col2] = true
                    append!(JJfoo, repeat(findall(foo[sd.solve_mask]), NTFC))
                end
            end
        end
        # Construct the sparse matrix
        sd.CiSc .= sparse(IIfoo, JJfoo, VVfoo, NTFC * sd.NU, size(sd.J, 1))
    end

    if changed
        # cached lu is no longer valid, since active columns have changed
        sd.luJ[] = nothing
    end

    return sd
end

"""
    make_BI(J, II)

Prepares the `BI` array for the solver data. Called from the constructor of
`StackedTimeSolverData`.

"""
function make_BI(JMAT::SparseMatrixCSC{Float64,Int64}, II::AbstractVector{<:AbstractVector{Int64}})
    # Computes the set of indexes in JMAT.nzval corresponding to blocks of equations in II

    # Bar is the inverse map of II, i.e. Bar[j] = i <=> II[i] contains j
    local Bar = zeros(Int64, size(JMAT, 1))
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
            push!(BI[ Bar[row] ], j)
        end
    end
    return BI
end

"""
    StackedTimeSolverData(model, plan, fctype)

"""
function StackedTimeSolverData(m::Model, p::Plan, fctype::FCType)

    # NOTE: we do not verify the steady state, just make sure it's been assigned
    if fctype ∈ (fclevel, fcslope) && !issssolved(m)
        throw(ArgumentError("Steady state must be solved for `$(fctype)`."))
    end

    NT = length(p.range)
    init = 1:m.maxlag
    term = NT - m.maxlead + 1:NT
    sim = m.maxlag + 1:NT - m.maxlead
    NTFC = length(term)
    NTSIM = length(sim)
    NTIC = length(init)

    if (fctype == fcnatural) && (length(sim) + length(init) < 2)
        throw(ArgumentError("Simulation must include at least 2 periods for `fcnatural`."))
    end

    # @assert NTIC+NTSIM+NTFC == NT
    # @assert isempty((collect(init)∩collect(sim))∪(collect(init)∩collect(sim))∪(collect(sim)∩collect(term)))

    unknowns = ModelBaseEcon.allvars(m)
    nunknowns = length(unknowns)

    nvars = ModelBaseEcon.nvariables(m)
    nshks = ModelBaseEcon.nshocks(m)
    nauxs = ModelBaseEcon.nauxvars(m)

    equations = ModelBaseEcon.alleqns(m)
    nequations = length(equations)

    # LinearIndices used for indexing the columns of the global matrix
    LI = LinearIndices((p.range, 1:nunknowns))

    # Initialize empty arrays
    TT = Vector{Int64}[] # the time indexes of the block, used when updating values in global_RJ
    II = Vector{Int64}[] # the row indexes
    JJ = Vector{Int64}[] # the the column indexes

    # Prep the Jacobian matrix
    neq::Int64 = 0 # running counter of equations added to matrix
    # Model equations are the same for each sim period, just shifted according to t
    Jblock = [ti + NT * (vi - 1) for eqn in equations for (ti, vi) in eqn.vinds]
    Iblock = [i for (i, eqn) in enumerate(equations) for _ in eqn.vinds]
    Tblock = -m.maxlag:m.maxlead
    @timer for t in sim
        push!(TT, t .+ Tblock)
        push!(II, neq .+ Iblock)
        push!(JJ, t .+ Jblock)
        neq += nequations
    end
    # Construct the
    @timer begin
        I = vcat(II...)
        J = vcat(JJ...)
        JMAT = sparse(I, J, fill(NaN64, size(I)), NTSIM * nequations, NT * nunknowns)
    end

    # We no longer need the exact indexes of all non-zero entires in the Jacobian matrix.
    # We do however need the set of equation indexes for each sim period
    @timer foreach(sort! ∘ unique!, II)

    # BI holds the indexes in JMAT.nzval for each block of equations
    BI = make_BI(JMAT, II)  # same as the two lines above, but faster

    # We also need the times of the final conditions
    push!(TT, term)

    # 
    exog_mask::Vector{Bool} = falses(nunknowns * NT)
    # Initial conditions are set as exogenous values
    exog_mask[vec(LI[init,:])] .= true
    # The exogenous values during sim are set in update_plan!() call below.

    # Final conditions are complicated
    fc_mask::Vector{Bool} = falses(nunknowns * NT)
    fc_mask[vec(LI[term,:])] .= true

    # The solve_mask is redundant. We pre-compute and store it for speed
    solve_mask = .!(fc_mask .| exog_mask)

    sd = StackedTimeSolverData(NT, nvars, nshks, nunknowns, fctype, TT, II, JJ, BI, JMAT,
                             sparse([], [], Float64[], NTFC * nunknowns, size(JMAT, 1)),
                             m.sstate.values, m.evaldata, exog_mask, fc_mask, solve_mask,
                             Ref{Any}(nothing))

    return update_plan!(sd, m, p; changed=true)
end

@inline function assign_exog_data!(x::AbstractArray{Float64,2}, exog::AbstractArray{Float64,2}, sd::StackedTimeSolverData)
    # @assert size(x,1) == size(exog,1) == sd.NT
    x[sd.exog_mask] = exog[sd.exog_mask]
    assign_final_condition!(x, exog, sd, Val(sd.FC))
    return x
end

@inline function assign_final_condition!(x::AbstractArray{Float64,2}, exog::AbstractArray{Float64,2}, sd::StackedTimeSolverData, ::Val{fcgiven})
    x[sd.TT[end],:] = exog[sd.TT[end],:]
    return x
end

@inline function assign_final_condition!(x::AbstractArray{Float64,2}, ::AbstractArray{Float64,2}, sd::StackedTimeSolverData, ::Val{fclevel})
    for t in sd.TT[end]
        x[t,:] = sd.SSV[1:2:end]
    end
    return x
end

@inline function assign_final_condition!(x::AbstractArray{Float64,2}, ::AbstractArray{Float64,2}, sd::StackedTimeSolverData, ::Val{fcrate})
    for t in sd.TT[end]
        x[t,:] = x[t-1,:] .+ sd.SSV[2:2:end]
    end
    return x
end

@inline function assign_final_condition!(x::AbstractArray{Float64,2}, ::AbstractArray{Float64,2}, sd::StackedTimeSolverData, ::Val{fcnatural})
    last_T = sd.TT[end][1] - 1
    SLP = x[last_T, : ] .- x[last_T - 1, :]
    Vinds = 1:sd.NV  # indices of variables
    Sinds = sd.NV + 1:sd.NV + sd.NS  # indices of shocks
    Ainds = sd.NV + sd.NS + 1:sd.NU  # indices of aux variables
    for t in sd.TT[end]
        x[t,Vinds] = x[t - 1,Vinds] .+ SLP[Vinds]
        x[t,Sinds] .= 0.0  # shocks are always 0 in final conditions using steady state
        x[t,Ainds] = x[t - 1,Ainds] .+ SLP[Ainds]
    end
    return x
end

function global_R!(res::AbstractArray{Float64,1}, point::AbstractArray{Float64}, exog_data::AbstractArray{Float64}, sd::StackedTimeSolverData)
    point = reshape(point, sd.NT, sd.NU)
    exog_data = reshape(point, sd.NT, sd.NU)
    @timer "global_R!" begin
        for (ii, tt) in zip(sd.II, sd.TT)
            eval_R!(view(res, ii), point[tt,:], sd.evaldata)
        end
    end
    return res
end

function global_RJ(point::AbstractArray{Float64}, exog_data::AbstractArray{Float64}, sd::StackedTimeSolverData)
    nunknowns = sd.NU
    point = reshape(point, sd.NT, nunknowns)
    exog_data = reshape(exog_data, sd.NT, nunknowns)
    JAC = sd.J
    RES = Vector{Float64}(undef, size(JAC, 1))
    # Model equations @ [1] to [end-1]
    haveJ = isa(sd.evaldata, ModelBaseEcon.LinearizedModelEvaluationData) && !any(isnan.(JAC.nzval))
    haveLU = sd.luJ[] !== nothing
    if haveJ
        # update only RES
        @timer "globalRJ/evalR!" for i = 1:length(sd.BI)
            eval_R!(view(RES, sd.II[i]), point[sd.TT[i],:], sd.evaldata)
        end
    else
        # update both RES and JAC
        @timer "globalRJ/evalRJ" for i = 1:length(sd.BI)
            R, J = eval_RJ(point[sd.TT[i],:], sd.evaldata)
            RES[sd.II[i]] .= R
            JAC.nzval[sd.BI[i]] .= J.nzval
        end
        haveLU = false
    end
    if !haveLU
        if nnz(sd.CiSc) > 0
            JJ = JAC[:, sd.solve_mask] - JAC[:, sd.fc_mask] * sd.CiSc
        else
            JJ = JAC[:, sd.solve_mask]
        end
        try
            # compute lu decomposition of the active part of J and cache it.
            @timer "LU decomposition" sd.luJ[] = lu(JJ)
        catch e
            if e isa SingularException
                error("The system is underdetermined with the given set of equations and final conditions.")
            end
            rethrow()
        end
    end
    return RES, sd.luJ[]
end

function assign_update_step!(x::AbstractArray{Float64}, λ::Float64, Δx::AbstractArray{Float64}, sd::StackedTimeSolverData)
    x[sd.solve_mask] .+= λ .* Δx
    if nnz(sd.CiSc) > 0
        x[sd.fc_mask] .-= λ .* (sd.CiSc * Δx)
        # assign_final_condition!(x, zeros(0,0), sd, Val(fcslope))
    end
    return x
end
