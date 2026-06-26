##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# ----------------------------------------------------------------------
# First-order system assembly + solve.
#
# Builds the linearised system FWD*x[t+1] + BCK*x[t] + EX*e[t] = 0 from the
# model's Jacobian at the steady state, runs the QZ decomposition, and
# assembles the decision-rule matrices (RbyZbb, MAT) that drive the
# per-period recursion in simulate.jl.
#
# Notes on the assembly:
#   - the Jacobian is built directly here by evaluating each equation's
#     gradient at the SS point via `eqn.eval_RJ!` (linear-model-exact). See
#     `_assemble_jacobian`.
#   - the global variable index is the unified `[vars; shocks]` column space
#     (same as SimData/SimPlan), so first-order output drops straight into
#     the simulation array layout.
# ----------------------------------------------------------------------

const _VarLag = Tuple{Symbol, Int}

"""
Classifies the model's variables into backward-looking (`bck`),
forward-looking (`fwd`) and exogenous/shock (`ex`) first-order variables. A
variable that appears at both a lag and a lead is *both* a bck and a fwd
variable (a mixed variable).

Global indexing (`vi`) is over the unified `[vars; shocks]` space, matching
the SimData/SimPlan column order.
"""
struct VarMaps
    vi::Dict{Symbol, Int}            # var/shock name -> global column index
    nfwd::Int
    nbck::Int
    nex::Int
    oex::Int                         # offset of the ex class = nbck + nfwd
    bck_vars::Vector{_VarLag}
    fwd_vars::Vector{_VarLag}
    ex_vars::Vector{_VarLag}
    bck_inds::Dict{_VarLag, Int}
    fwd_inds::Dict{_VarLag, Int}     # indices continue from nbck
    ex_inds::Dict{_VarLag, Int}      # indices restart from 1
    inds_map::Vector{Tuple{Int, Int}}  # fo-index -> (global col, lag)
end

function VarMaps(model::ModelBaseEcon.CompiledModel)
    def = model.defs
    n_var = length(def.vars)

    # global index over [vars; shocks]
    vi = Dict{Symbol, Int}()
    for (i, v) in pairs(def.vars);   vi[v.name] = i;          end
    for (i, s) in pairs(def.shocks); vi[s.name] = n_var + i;  end
    shock_names = Set(s.name for s in def.shocks)

    # per-name min lag / max lead across all equations
    lag_of  = Dict{Symbol, Int}()
    lead_of = Dict{Symbol, Int}()
    for eqn in model.eqns
        for ref in eqn.tsrefs
            lo = get(lag_of,  ref.name, 0)
            hi = get(lead_of, ref.name, 0)
            lag_of[ref.name]  = min(lo, ref.offset)
            lead_of[ref.name] = max(hi, ref.offset)
        end
    end

    fwd_vars = _VarLag[]
    bck_vars = _VarLag[]
    ex_vars  = _VarLag[]

    # exogenous/shock variables first (one entry per offset they appear at),
    # in declaration order. Shocks are a distinct list, and we push
    # `(var, tt) for tt = lags:leads`.
    for s in def.shocks
        var = s.name
        lags  = get(lag_of,  var, 0)
        leads = get(lead_of, var, 0)
        for tt in lags:leads
            push!(ex_vars, (var, tt))
        end
    end

    # genuine variables -> bck / fwd
    for v in def.vars
        var = v.name
        lags  = get(lag_of,  var, 0)
        leads = get(lead_of, var, 0)
        if lags == 0 && leads == 0
            push!(bck_vars, (var, 0))
        else
            for tt in 1:(-lags)
                push!(bck_vars, (var, 1 - tt))
            end
            for tt in 1:leads
                push!(fwd_vars, (var, tt - 1))
            end
        end
    end

    nbck = length(bck_vars)
    nfwd = length(fwd_vars)
    nex  = length(ex_vars)
    oex  = nbck + nfwd

    bck_inds = Dict{_VarLag, Int}(key => i for (i, key) in enumerate(bck_vars))
    fwd_inds = Dict{_VarLag, Int}(key => nbck + i for (i, key) in enumerate(fwd_vars))
    ex_inds  = Dict{_VarLag, Int}(key => i for (i, key) in enumerate(ex_vars))

    inds_map = Vector{Tuple{Int, Int}}(undef, nbck + nfwd + nex)
    for ((v, t), i) in bck_inds; inds_map[i] = (vi[v], t); end
    for ((v, t), i) in fwd_inds; inds_map[i] = (vi[v], t); end
    for ((v, t), i) in ex_inds;  inds_map[oex + i] = (vi[v], t); end

    return VarMaps(vi, nfwd, nbck, nex, oex,
                   bck_vars, fwd_vars, ex_vars,
                   bck_inds, fwd_inds, ex_inds, inds_map)
end

# ----------------------------------------------------------------------
# Build the model Jacobian at the steady state.
#
# Returns a SparseMatrixCSC with `n_eq` rows and `n_col * (1+maxlag+maxlead)`
# columns, where the column for (global var index `vno`, time offset `tt`)
# is `(vno-1)*(1+maxlag+maxlead) + (tt+maxlag) + 1` - the same column layout
# `fill_fosystem!` decodes via `divrem(col-1, 1+maxlag+maxlead)`.
#
# x_ss_global is indexed over [vars; shocks] (shocks' SS = 0).
# ----------------------------------------------------------------------
function _assemble_jacobian(model::ModelBaseEcon.CompiledModel,
                            x_ss_global::AbstractVector{Float64},
                            p::Vector{Float64},
                            maxlag::Int, maxlead::Int)
    def = model.defs
    n_var = length(def.vars)
    n_shock = length(def.shocks)
    n_col = n_var + n_shock
    n_eq = length(model.eqns)
    win = 1 + maxlag + maxlead

    var_index = Dict{Symbol, Int}()
    for (i, v) in pairs(def.vars);   var_index[v.name] = i;          end
    for (i, s) in pairs(def.shocks); var_index[s.name] = n_var + i;  end

    I = Int[]; Jc = Int[]; V = Float64[]
    grad = Float64[]
    for (eqind, eqn) in pairs(model.eqns)
        if length(grad) != eqn.n_x
            resize!(grad, eqn.n_x)
        end
        # per-slot SS point: var -> its SS value, shock -> 0
        x_eqn = Vector{Float64}(undef, eqn.n_x)
        for (k, ref) in pairs(eqn.tsrefs)
            gi = var_index[ref.name]
            x_eqn[k] = x_ss_global[gi]
        end
        eqn.eval_RJ!(grad, x_eqn, p)
        for (k, ref) in pairs(eqn.tsrefs)
            g = grad[k]
            g == 0.0 && continue
            gi = var_index[ref.name]
            col = (gi - 1) * win + (ref.offset + maxlag) + 1
            push!(I, eqind); push!(Jc, col); push!(V, g)
        end
    end
    return SparseArrays.sparse(I, Jc, V, n_eq, n_col * win)
end

# ----------------------------------------------------------------------
# FirstOrderSystem - FWD, BCK, EX matrices.
# ----------------------------------------------------------------------
struct FirstOrderSystem
    FWD::Matrix{Float64}
    BCK::Matrix{Float64}
    EX::Matrix{Float64}
end

function FirstOrderSystem(JAC::SparseArrays.SparseMatrixCSC,
                          model::ModelBaseEcon.CompiledModel, vm::VarMaps,
                          maxlag::Int, maxlead::Int)
    dim = vm.nbck + vm.nfwd
    sys = FirstOrderSystem(zeros(dim, dim), zeros(dim, dim), zeros(dim, vm.nex))
    return fill_fosystem!(sys, JAC, model, vm, maxlag, maxlead)
end

function fill_fosystem!(sys::FirstOrderSystem, JAC::SparseArrays.SparseMatrixCSC,
                        model::ModelBaseEcon.CompiledModel, vm::VarMaps,
                        maxlag::Int, maxlead::Int)
    FWD = fill!(sys.FWD, 0.0)
    BCK = fill!(sys.BCK, 0.0)
    EX  = fill!(sys.EX, 0.0)
    win = 1 + maxlag + maxlead

    n_var = length(model.defs.vars)
    # inverse global-index -> name
    name_of = Vector{Symbol}(undef, n_var + length(model.defs.shocks))
    for (i, v) in pairs(model.defs.vars);   name_of[i] = v.name;          end
    for (i, s) in pairs(model.defs.shocks); name_of[n_var + i] = s.name;  end

    for (eqind, col, val) in zip(SparseArrays.findnz(JAC)...)
        (vno, tt) = divrem(col - 1, win)
        vno += 1
        tt -= maxlag
        var = name_of[vno]
        var_tt = (var, tt)
        ex_i = get(vm.ex_inds, var_tt, nothing)
        if ex_i !== nothing
            EX[eqind, ex_i] = val
            continue
        end
        if tt < 0
            bck_i = get(vm.bck_inds, (var, tt + 1), nothing)
            BCK[eqind, bck_i] = val
        elseif tt > 0
            fwd_i = get(vm.fwd_inds, (var, tt - 1), nothing)
            FWD[eqind, fwd_i] = val
        else # tt == 0
            bck_i = get(vm.bck_inds, (var, 0), nothing)
            if bck_i !== nothing
                FWD[eqind, bck_i] = val
            else
                fwd_i = get(vm.fwd_inds, (var, 0), nothing)
                BCK[eqind, fwd_i] = val
            end
        end
    end

    # auxiliary "link" rows tying multi-lag/lead copies + fwd<->bck cross-link
    eqn = length(model.eqns)
    for (var, tt) in vm.fwd_vars
        if tt == 0
            b_i = get(vm.bck_inds, (var, 0), nothing)
            if b_i !== nothing
                eqn += 1
                f_i = vm.fwd_inds[(var, 0)]
                FWD[eqn, b_i] = 1
                BCK[eqn, f_i] = -1
            end
        else
            eqn += 1
            FWD[eqn, vm.fwd_inds[(var, tt - 1)]] = 1
            BCK[eqn, vm.fwd_inds[(var, tt)]] = -1
        end
    end
    for (var, tt) in vm.bck_vars
        if tt == 0
            continue
        else
            eqn += 1
            FWD[eqn, vm.bck_inds[(var, tt)]] = 1
            BCK[eqn, vm.bck_inds[(var, tt + 1)]] = -1
        end
    end
    return sys
end

# ----------------------------------------------------------------------
# Blanchard-Kahn / decision rule.
# ----------------------------------------------------------------------
"""
Raised when the Blanchard-Kahn conditions cannot produce a unique stable
solution (the QZ block structure is not invertible as required). Carries
the eigenvalue counts for diagnosis.
"""
struct BlanchardKahnError <: Exception
    nbck::Int
    nfwd::Int
    msg::String
end
Base.showerror(io::IO, e::BlanchardKahnError) =
    print(io, "BlanchardKahnError(nbck=$(e.nbck), nfwd=$(e.nfwd)): ", e.msg)

function _decision_rule(qz::QZResult, EX::Matrix{Float64},
                        nbck::Int, nfwd::Int, nex::Int)
    oex = nbck + nfwd

    Zbb = LinearAlgebra.lu(qz.Z[1:nbck, 1:nbck])
    Zbf = qz.Z[1:nbck, nbck .+ (1:nfwd)]
    Zfb = qz.Z[nbck .+ (1:nfwd), 1:nbck]
    Zff = qz.Z[nbck .+ (1:nfwd), nbck .+ (1:nfwd)]

    Tbb = qz.T[1:nbck, 1:nbck]
    Tbf = qz.T[1:nbck, nbck .+ (1:nfwd)]
    Tff = LinearAlgebra.lu(qz.T[nbck .+ (1:nfwd), nbck .+ (1:nfwd)])

    Sbb = qz.S[1:nbck, 1:nbck]

    QEX = qz.Q' * EX

    R = vcat(-Tbb, Zff' * Zfb)

    TiXf = Tff \ QEX[nbck .+ (1:nfwd), :]

    MAT = zeros(nbck + nfwd, nbck + nfwd + nex)

    # backward-looking rows
    MAT[1:nbck, 1:nbck] = Sbb / Zbb
    MAT[1:nbck, oex .+ (1:nex)] = QEX[1:nbck, :] - (Tbf - Tbb * (Zbb \ Zbf)) * TiXf

    # forward-looking rows
    MAT[nbck .+ (1:nfwd), nbck .+ (1:nfwd)] = Zff'
    MAT[nbck .+ (1:nfwd), oex .+ (1:nex)] = TiXf

    return LinearAlgebra.rdiv!(R, Zbb), MAT
end

# ----------------------------------------------------------------------
# FirstOrderModel - the public solved model.
# ----------------------------------------------------------------------
"""
The solved first-order model. Carries the variable classification (`vm`),
the QZ result, the decision-rule matrices (`RbyZbb`, `MAT` with its
pre-factorised `MAT_n` empty-plan solver and `MAT_x` exogenous-input view),
the steady-state reference (`x_ss`, indexed over `[vars; shocks]`), and
the model + lag structure needed by `first_order_simulate` /
`first_order_shockdecomp`.
"""
struct FirstOrderModel
    model::ModelBaseEcon.CompiledModel
    vm::VarMaps
    sys::FirstOrderSystem
    qz::QZResult
    RbyZbb::Matrix{Float64}
    MAT::Matrix{Float64}
    MAT_n::LinearAlgebra.Factorization{Float64}
    MAT_x::SubArray{Float64, 2, Matrix{Float64}}
    x_ss::Vector{Float64}        # length n_var + n_shock ([vars; shocks])
    maxlag::Int
    maxlead::Int
end

"""
    first_order_solve(model, x_ss_vars; resolve_params=true) -> FirstOrderModel

Solve the linear rational-expectations model around the steady state
`x_ss_vars` (length `length(model.defs.vars)`, in solver space - i.e. the
`@log`-variable entries are `log(level)`). Builds the canonical
`FWD*x[t+1] + BCK*x[t] + EX*e[t] = 0` system from the model Jacobian at the
SS, runs QZ, and returns the decision rule.

`p` (resolved parameter values) is taken from the model's `param_layout`.
"""
function first_order_solve(model::ModelBaseEcon.CompiledModel,
                           x_ss_vars::AbstractVector{Float64})
    def = model.defs
    n_var = length(def.vars)
    n_shock = length(def.shocks)
    length(x_ss_vars) == n_var ||
        error("first_order_solve: x_ss has length $(length(x_ss_vars)), expected $n_var")

    # maxlag / maxlead from equation tsrefs
    maxlag = 0; maxlead = 0
    for eqn in model.eqns, ref in eqn.tsrefs
        maxlag  = max(maxlag, -ref.offset)
        maxlead = max(maxlead, ref.offset)
    end

    # SS over [vars; shocks] (shocks' SS = 0)
    x_ss = vcat(collect(Float64, x_ss_vars), zeros(n_shock))

    p = _resolve_param_values(def, model.param_layout)

    vm = VarMaps(model)
    JAC = _assemble_jacobian(model, x_ss, p, maxlag, maxlead)
    sys = FirstOrderSystem(JAC, model, vm, maxlag, maxlead)

    qz = run_qz(sys.FWD, sys.BCK, vm.nbck)

    local RbyZbb, MAT
    try
        RbyZbb, MAT = _decision_rule(qz, sys.EX, vm.nbck, vm.nfwd, vm.nex)
    catch err
        if err isa LinearAlgebra.SingularException
            throw(BlanchardKahnError(vm.nbck, vm.nfwd,
                "QZ block not invertible — Blanchard-Kahn conditions not met " *
                "(check stable/unstable eigenvalue counts vs forward-looking variables)"))
        end
        rethrow()
    end

    MAT_n = LinearAlgebra.lu(MAT[:, 1:vm.oex])
    MAT_x = view(MAT, :, vm.oex .+ (1:vm.nex))

    return FirstOrderModel(model, vm, sys, qz, RbyZbb, MAT, MAT_n, MAT_x,
                           x_ss, maxlag, maxlead)
end

# Local param resolver - mirror of Linearize._resolve_param_values
# (ModelBaseEcon/src/linearize.jl). Duplicated rather than imported because
# that helper is internal to the ModelBaseEcon Linearize module.
function _resolve_param_values(def::IR.ModelDef,
                               layout::Vector{Symbolic.ParamRef})
    by_name = Dict{Symbol, IR.ParamDecl}()
    for p in def.params; by_name[p.name] = p; end
    vals = Vector{Float64}(undef, length(layout))
    for (i, ref) in pairs(layout)
        p = by_name[ref.name]
        if p.kind === IR.PARAM_SCALAR
            vals[i] = Float64(p.value)
        elseif p.kind === IR.PARAM_ARRAY
            vals[i] = Float64(p.value[ref.index])
        else
            error("first_order_solve: root layout should hold scalar/array params, got $(p.kind)")
        end
    end
    return vals
end
