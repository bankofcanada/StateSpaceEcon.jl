# ----------------------------------------------------------------------
# SteadyStateProblem - precomputed wiring from a model to the SS Newton
#
# At steady state every variable is time-invariant: x[t-k] = x[t] = x[t+k]
# = x_ss[i]. Shocks are zero. So for each equation we evaluate
#     R_i(x_ss, p) = eq.eval_resid(x_eqn, p)
# where x_eqn[k] = x_ss[var_idx[k]] if tsrefs[k].name is a variable, else 0.0.
#
# The Jacobian row in the global `n_eq x n_var` system is built by
# summing gradient entries across offsets of the same variable.
# ----------------------------------------------------------------------

"""
Mapping from per-equation `x` slot to global variable index. `nothing`
means this slot is a shock (held at zero in SS).
"""
const SlotMap = Vector{Union{Int, Nothing}}

struct SteadyStateProblem
    model::ModelBaseEcon.CompiledModel
    n_var::Int
    n_eq::Int                                 # auto-derived equation count
    n_ss_user::Int                            # user-supplied SS equation count
    var_index::Dict{Symbol, Int}              # variable name -> index in x_ss
    slot_maps::Vector{SlotMap}                # per-(auto)equation: x slot -> var index
    ss_user_slot_maps::Vector{SlotMap}        # per-user-SS-eqn: x slot -> var index
    param_values::Vector{Float64}             # flat root-param vector
end

"""
    n_total_eq(prob) -> Int

Total row count in the augmented SS system (auto-derived equations +
user-supplied `@steadystate` constraints). When this exceeds `n_var`,
the system is over-determined and `sssolve!` drops to a least-squares
step.
"""
n_total_eq(prob::SteadyStateProblem) = prob.n_eq + prob.n_ss_user

"""
    SteadyStateProblem(compiled::CompiledModel)

Build the precomputed mapping. Parameter values are read from the
underlying `ModelDef` (via `compiled.defs.params`) and resolved through
the link table. `compiled.param_layout` defines the canonical order.
"""
function SteadyStateProblem(compiled::ModelBaseEcon.CompiledModel)
    def = compiled.defs
    n_var = length(def.vars)
    n_eq = length(compiled)

    var_index = Dict{Symbol, Int}()
    for (i, v) in pairs(def.vars)
        var_index[v.name] = i
    end
    shock_names = Set(s.name for s in def.shocks)

    slot_maps = SlotMap[]
    for eqn in compiled.eqns
        sm = SlotMap(undef, eqn.n_x)
        for (k, ref) in pairs(eqn.tsrefs)
            if haskey(var_index, ref.name)
                sm[k] = var_index[ref.name]
            elseif ref.name in shock_names
                sm[k] = nothing
            else
                error("steady-state: equation references unknown name `$(ref.name)`")
            end
        end
        push!(slot_maps, sm)
    end

    # User-supplied @steadystate constraints. The kernel slot order
    # matches the same tsrefs convention as dynamic equations.
    ss_user_slot_maps = SlotMap[]
    for sseq in compiled.ss_eqns
        sm = SlotMap(undef, sseq.eqn.n_x)
        for (k, ref) in pairs(sseq.eqn.tsrefs)
            if haskey(var_index, ref.name)
                sm[k] = var_index[ref.name]
            elseif ref.name in shock_names
                sm[k] = nothing
            else
                error("steady-state: @steadystate equation $(sseq.name) " *
                      "references unknown name `$(ref.name)`")
            end
        end
        push!(ss_user_slot_maps, sm)
    end
    n_ss_user = length(ss_user_slot_maps)

    param_values = _resolve_param_values(def, compiled.param_layout)

    total_eq = n_eq + n_ss_user
    if total_eq < n_var
        error("steady-state: under-determined system (n_eq=$n_eq, n_ss_user=$n_ss_user, " *
              "n_var=$n_var). Add @steadystate constraints to close the system.")
    end

    return SteadyStateProblem(compiled, n_var, n_eq, n_ss_user, var_index,
                               slot_maps, ss_user_slot_maps, param_values)
end

# Resolve every root parameter to a Float64 by walking the link table
# numerically. Linked params have already been substituted away from
# the root layout (they don't appear), but their *values* don't matter
# for SS because the kernel residual was rewritten to reference root
# params only. We need the numeric values of the root params here.
function _resolve_param_values(def::IR.ModelDef,
                                layout::Vector{Symbolic.ParamRef})
    by_name = Dict{Symbol, IR.ParamDecl}()
    for p in def.params
        by_name[p.name] = p
    end
    vals = Vector{Float64}(undef, length(layout))
    for (i, ref) in pairs(layout)
        p = by_name[ref.name]
        if p.kind === IR.PARAM_SCALAR
            vals[i] = Float64(p.value)
        elseif p.kind === IR.PARAM_ARRAY
            vals[i] = Float64(p.value[ref.index])
        else
            error("root layout should only hold scalar/array params, got $(p.kind)")
        end
    end
    return vals
end
