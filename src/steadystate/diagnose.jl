# Steady-state convergence diagnostics.
#
# Returns a structured diagnosis (per-equation residuals + row slopes,
# Jacobian condition number, the worst-residual equation index, and
# near-singular row indices).

"""
    SteadyStateDiagnosis

Result of [`diagnose_sstate`](@ref). Fields:

- `x::Vector{Float64}` - point at which the diagnostic was evaluated.
- `residuals::Vector{Float64}` - per-equation SS residual.
- `slopes::Vector{Float64}` - per-equation row slope `‖row_i(J)‖∞`.
- `jacobian_cond::Float64` - `cond(J)` of the SS Jacobian, or `NaN`
  when `n_var > 1000` (dense `cond` is `O(n^3)`).
- `worst_residual_eqn::Int` - `argmax(abs.(residuals))`.
- `near_singular_rows::Vector{Int}` - equation indices whose row in
  the SS Jacobian has `‖row‖∞ < row_tol` (default `1e-10`).
"""
struct SteadyStateDiagnosis
    x::Vector{Float64}
    residuals::Vector{Float64}
    slopes::Vector{Float64}
    jacobian_cond::Float64
    worst_residual_eqn::Int
    near_singular_rows::Vector{Int}
end

# Dense `cond` is O(n^3); skip past this size to keep the diagnostic
# itself from blowing up on large models.
const _COND_DENSE_LIMIT = 1000

"""
    diagnose_sstate(prob::SteadyStateProblem, x_attempt=nothing;
                    row_tol=1e-10)

Evaluate diagnostic statistics on the steady-state system at
`x_attempt`. When `x_attempt === nothing`, defaults to `ones(n_var)`
(the same default `sssolve!` uses for its initial guess).
"""
function diagnose_sstate(prob::SteadyStateProblem,
                          x_attempt::Union{AbstractVector{<:Real}, Nothing} = nothing;
                          row_tol::Float64 = 1e-10)
    n = prob.n_var
    x = x_attempt === nothing ? ones(Float64, n) : collect(Float64, x_attempt)
    length(x) == n || error("diagnose_sstate: x_attempt length $(length(x)) != n_var $n")

    m = n_total_eq(prob)
    R = Vector{Float64}(undef, m)
    J = Matrix{Float64}(undef, m, n)
    ss_RJ!(R, J, x, prob)

    slopes = [norm(@view(J[i, :]), Inf) for i in 1:m]
    near_singular_rows = [i for i in 1:m if slopes[i] < row_tol]
    worst = argmax(abs.(R))

    jcond = if n > _COND_DENSE_LIMIT
        NaN
    else
        try
            cond(J)
        catch
            NaN
        end
    end

    return SteadyStateDiagnosis(x, R, slopes, jcond, worst, near_singular_rows)
end

function Base.show(io::IO, d::SteadyStateDiagnosis)
    n = length(d.residuals)
    println(io, "SteadyStateDiagnosis: $n equations")
    println(io, "  jacobian_cond      = ", d.jacobian_cond)
    println(io, "  worst_residual_eqn = E$(d.worst_residual_eqn) ",
                  "(residual=", d.residuals[d.worst_residual_eqn], ")")
    if isempty(d.near_singular_rows)
        println(io, "  near_singular_rows = (none)")
    else
        println(io, "  near_singular_rows = ", d.near_singular_rows)
    end
    # Ranked table by |residual| desc.
    order = sortperm(abs.(d.residuals); rev = true)
    println(io, "  ┌──────┬──────────────────┬──────────────────┐")
    println(io, "  │  eq  │     residual     │   row slope ∞    │")
    println(io, "  ├──────┼──────────────────┼──────────────────┤")
    show_n = min(n, 10)
    for k in 1:show_n
        i = order[k]
        println(io, "  │ ",
                lpad("E$i", 4), " │ ",
                lpad(string(d.residuals[i]), 16), " │ ",
                lpad(string(d.slopes[i]), 16), " │")
    end
    if n > show_n
        println(io, "  │  …   │   ($(n - show_n) more rows hidden)         │")
    end
    print(io,   "  └──────┴──────────────────┴──────────────────┘")
end

_diagnose_for_sssolve(prob::SteadyStateProblem, x::AbstractVector) =
    diagnose_sstate(prob, x)
