module ShockDecomp

# ----------------------------------------------------------------------
# Stacked-time shock decomposition, expressed on the `SimPlan` / `SimData`
# surface.
#
# --- The algorithm ---
#
# Given a `control` solution and a `shocked` solution of the same
# stacked-time system, attribute the difference `delta = shocked -
# control`, cell by cell, to each exogenous source: the initial
# conditions, each shock column, and (when there are leads) the terminal
# conditions. The attribution is exact to first order; the residual of
# the linear approximation is reported as a `nonlinear` column.
#
# Index everything over the *full data matrix*: `n_rows x n_col`, where
# `n_rows = maxlag + T + maxlead` and the n_col unified columns are
# `[vars; shocks]`. Flatten variable-major: point(row,col) =
# (col-1)*n_rows + row. The stacked residual has, per simulation period
# t = 1..T and equation i, a row (t-1)*n_eq + i.
#
#   * J_full - d(residual)/d(every data-matrix cell), size
#              (n_eq.T) x (n_col.n_rows), evaluated at `control`.
#   * endogenous columns - variable columns (1..n_var) at the T interior
#     rows (maxlag+1 .. maxlag+T): the points the solver solves for.
#   * J_endo - J_full restricted to those columns, square (n_eq.T).
#
# Split the remaining (exogenous) cells into source groups g: `init`
# (all columns, rows 1..maxlag), each shock column over the interior
# rows, `term` (all columns, the maxlead terminal rows). For group g let
# d_g hold `delta` on g's cells, 0 elsewhere. The endogenous response
# decomposed by source solves, column by column,
#
#   J_endo . S = -J_full . D            D = hcat(d_g for each group g)
#
# The endogenous rows of S give each variable cell's contribution from
# each source; each source's own cells get `delta` directly. The
# leftover `delta - Sum_g contribution_g` is the `nonlinear` column.
# ----------------------------------------------------------------------

using ModelBaseEcon
using LinearAlgebra: norm, lu
using SparseArrays: sparse

using ..Plans
using ..Plans: SimData, SimPlan

export ShockDecompResult, shock_decomp

# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

"""
    ShockDecompResult

Output of [`shock_decomp`](@ref).

Fields:
* `source_names :: Vector{Symbol}` - the decomposition source groups, in
  column order: `:init`, then one per shock/exogenous column, then
  `:term` (only when the model has leads), then `:nonlinear`.
* `contrib :: Dict{Symbol, Matrix{Float64}}` - per endogenous variable
  name, a `(maxlag + T) x n_source` matrix. Row `maxlag + t` is
  simulation period `t`; column `k` is the contribution of
  `source_names[k]`.
* `control :: SimData` - the supplied control solution (copied).
* `shocked :: SimData` - the supplied shocked solution (copied).

For every endogenous variable `v` and simulation period `t`,
`sum(contrib[v][maxlag+t, :]) ≈ shocked[v,t] - control[v,t]`.
"""
struct ShockDecompResult
    source_names::Vector{Symbol}
    contrib::Dict{Symbol,Matrix{Float64}}
    control::SimData
    shocked::SimData
end

# ----------------------------------------------------------------------
# Jacobian assembly at the control solution
# ----------------------------------------------------------------------

# Flat index of data-matrix cell (row, col): variable-major over the
# full (n_rows x n_col) matrix.
@inline _cellidx(row::Int, col::Int, n_rows::Int) = (col - 1) * n_rows + row

# Gather an equation's input vector from a data matrix (solver space).
@inline function _gather(M::AbstractMatrix{Float64}, sm, t::Int, maxlag::Int)
    n = length(sm)
    x = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        col, off = sm[k]
        x[k] = M[t + off + maxlag, col]
    end
    return x
end

# Build the full Jacobian: rows = stacked residuals (n_eq.T), cols =
# data-matrix cells (n_col.n_rows), evaluated at `M`. Every point an
# equation reads - interior, initial, or terminal - gets a real column,
# since boundary conditions are decomposition sources.
function _full_jacobian(plan::SimPlan, M::AbstractMatrix{Float64})
    T, n_eq = plan.T, plan.n_eq
    maxlag, maxlead = plan.maxlag, plan.maxlead
    n_rows = maxlag + T + maxlead
    p = plan.param_values
    Is = Int[]; Js = Int[]; Vs = Float64[]
    grad_buf = Float64[]
    @inbounds for t in 1:T
        for i in 1:n_eq
            eqn = plan.model[i]
            sm = plan.slot_maps[i]
            if length(grad_buf) != eqn.n_x
                resize!(grad_buf, eqn.n_x)
            end
            x = _gather(M, sm, t, maxlag)
            eqn.eval_RJ!(grad_buf, x, p)
            resid_row = (t - 1) * n_eq + i
            for (k, (col, off)) in pairs(sm)
                cell_row = t + off + maxlag        # row into the data matrix
                push!(Is, resid_row)
                push!(Js, _cellidx(cell_row, col, n_rows))
                push!(Vs, grad_buf[k])
            end
        end
    end
    return sparse(Is, Js, Vs, T * n_eq, n_rows * (plan.n_col))
end

# ----------------------------------------------------------------------
# The decomposition
# ----------------------------------------------------------------------

"""
    shock_decomp(plan::SimPlan, control::SimData, shocked::SimData;
                 tol=1e-9, verbose=false) -> ShockDecompResult

Decompose the difference between a `shocked` and a `control` solution of
the same stacked-time system into the contributions of the initial
conditions, each shock, and (when the model has leads) the terminal
conditions.

`plan` must have the default partition (every variable endogenous, every
shock exogenous) - this is the partition `SimPlan(model, T)` produces.
`control` and `shocked` must each be a solution: their residual is
checked and a warning issued if it exceeds `tol` (the algorithm assumes
both solve the model, but does not enforce it).

See [`ShockDecompResult`](@ref) for the output layout.
"""
function shock_decomp(plan::SimPlan, control::SimData, shocked::SimData;
                      tol::Float64 = 1e-9, verbose::Bool = false)
    plan.model === control.model === shocked.model ||
        error("shock_decomp: plan, control and shocked refer to different models")
    all(plan.is_unknown[1:plan.n_var]) && !any(plan.is_unknown[plan.n_var+1:end]) ||
        error("shock_decomp: plan must have the default partition " *
              "(all variables endogenous, all shocks exogenous)")

    T, n_eq = plan.T, plan.n_eq
    n_var, n_shock, n_col = plan.n_var, plan.n_shock, plan.n_col
    maxlag, maxlead = plan.maxlag, plan.maxlead
    n_rows = maxlag + T + maxlead

    size(control.values) == (n_rows, n_col) ||
        error("shock_decomp: control has $(size(control.values)) values, " *
              "expected $((n_rows, n_col))")
    size(shocked.values) == (n_rows, n_col) ||
        error("shock_decomp: shocked has $(size(shocked.values)) values, " *
              "expected $((n_rows, n_col))")

    # --- Residual checks: both inputs should solve the model. ---
    rc = _residual_norm(plan, control.values)
    rs = _residual_norm(plan, shocked.values)
    verbose && @info "shock_decomp residuals" control = rc shocked = rs
    rc > tol && @warn "shock_decomp: control is not a solution" residual = rc
    rs > tol && @warn "shock_decomp: shocked is not a solution" residual = rs

    # --- delta = shocked - control over the full data matrix. ---
    delta = shocked.values .- control.values

    # --- Jacobian at the control solution. ---
    J_full = _full_jacobian(plan, control.values)

    # Endogenous columns: variable columns at the T interior rows.
    endo_cols = Int[_cellidx(maxlag + t, v, n_rows) for v in 1:n_var for t in 1:T]
    J_endo = J_full[:, endo_cols]

    # --- Source groups, in output column order. ---
    # Columns are emitted in the order `init`, `term`, then one per shock,
    # then `nonlinear` - and `term` is always included even for an all-lag
    # model (its column is then 0).
    #   `init` - all columns, rows 1..maxlag.
    #   `term` - all columns, the maxlead terminal rows (empty if none).
    #   one group per shock column - interior rows.
    source_names = Symbol[]
    group_cells = Vector{Vector{Int}}()

    push!(source_names, :init)
    init_cells = Int[]
    for col in 1:n_col, r in 1:maxlag
        push!(init_cells, _cellidx(r, col, n_rows))
    end
    push!(group_cells, init_cells)

    push!(source_names, :term)
    term_cells = Int[]
    for col in 1:n_col, r in (maxlag + T + 1):n_rows
        push!(term_cells, _cellidx(r, col, n_rows))
    end
    push!(group_cells, term_cells)
    term_group = 2                                # `term` is group 2

    shock_syms = [s.name for s in plan.model.defs.shocks]
    for (si, sname) in enumerate(shock_syms)
        col = n_var + si
        cells = Int[_cellidx(maxlag + t, col, n_rows) for t in 1:T]
        push!(source_names, sname)
        push!(group_cells, cells)
    end

    n_group = length(group_cells)

    # --- D: one column per source group, holding delta on its cells. ---
    flat_delta = vec(delta)                       # data-matrix cell index -> Δ
    D = zeros(n_rows * n_col, n_group)
    for (g, cells) in pairs(group_cells)
        for c in cells
            D[c, g] = flat_delta[c]
        end
    end

    # --- Endogenous response: J_endo . S = -J_full . D. ---
    RHS = Matrix(J_full * D)                      # (n_eq.T) x n_group
    F = lu(J_endo)
    S = F \ (-RHS)                                # (n_eq.T) x n_group

    # --- Assemble per-variable contribution matrices. ---
    # Output columns: the source groups, then a trailing `:nonlinear`.
    out_names = vcat(source_names, :nonlinear)
    n_out = length(out_names)
    contrib = Dict{Symbol,Matrix{Float64}}()
    for (vi, v) in pairs(plan.model.defs.vars)
        M = zeros(n_rows, n_out)
        # Interior rows: the solved endogenous contributions from S.
        for t in 1:T
            srow = (vi - 1) * T + t               # row of S for (v,t)
            for g in 1:n_group
                M[maxlag + t, g] = S[srow, g]
            end
        end
        # Initial-condition rows: the whole delta goes to the `init`
        # source (column 1) - boundary data is its own source.
        for r in 1:maxlag
            M[r, 1] = delta[r, vi]
        end
        # Terminal rows: the whole delta goes to the `term` source.
        for r in (maxlag + T + 1):n_rows
            M[r, term_group] = delta[r, vi]
        end
        # `nonlinear` = delta - Sum contributions (the linear-approx error).
        for r in 1:n_rows
            s = 0.0
            for g in 1:n_group
                s += M[r, g]
            end
            M[r, n_out] = delta[r, vi] - s
        end
        contrib[v.name] = M
    end

    return ShockDecompResult(out_names, contrib, copy(control), copy(shocked))
end

# Inf-norm of the stacked residual of `M` (solver space) under `plan`.
function _residual_norm(plan::SimPlan, M::AbstractMatrix{Float64})
    T, n_eq, maxlag = plan.T, plan.n_eq, plan.maxlag
    p = plan.param_values
    worst = 0.0
    @inbounds for t in 1:T
        for i in 1:n_eq
            eqn = plan.model[i]
            x = _gather(M, plan.slot_maps[i], t, maxlag)
            r = eqn.eval_resid(x, p)
            worst = max(worst, abs(r))
        end
    end
    return worst
end

end # module ShockDecomp
