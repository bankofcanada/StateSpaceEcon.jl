module TimeSeriesEconExt

# StateSpaceEcon <-> upstream TimeSeriesEcon bridge.
#
# Consume the upstream TimeSeriesEcon.jl package as an optional dependency
# via this extension. Without TimeSeriesEcon loaded, the matrix API in
# Plans is primary. With TimeSeriesEcon loaded, callers can wrap SimData
# buffers as labelled MVTSeries views, adopt an existing MVTSeries into a
# SimData, or load a longbase CSV directly into an MVTSeries.

using StateSpaceEcon
using StateSpaceEcon.Plans
using StateSpaceEcon.Plans: SimData, load_longbase
import ModelBaseEcon
using TimeSeriesEcon
using TimeSeriesEcon: MIT, MVTSeries

# --- column order: vars (declaration order) then shocks. The matching
#     SimData/SimPlan code uses exactly this ordering, so we re-derive it
#     from the model rather than reading col_index back (which is a Dict
#     and not order-preserving).

function _ordered_colnames(model::ModelBaseEcon.CompiledModel)
    def = model.defs
    names = Vector{Symbol}(undef, length(def.vars) + length(def.shocks))
    for (i, v) in pairs(def.vars);   names[i] = v.name;                       end
    for (i, s) in pairs(def.shocks); names[length(def.vars) + i] = s.name;    end
    return names
end

# ----------------------------------------------------------------------
# (1) MVTSeries(sd::SimData, firstdate) - labelled view over sd.values.
#     The upstream inner constructor calls `view(values, :, ind)` per
#     column, so passing the SimData's raw Matrix gives an MVTSeries whose
#     per-column TSeries are views into the same buffer. Mutations through
#     the MVTSeries flow back to the SimData.
# ----------------------------------------------------------------------

function TimeSeriesEcon.MVTSeries(sd::SimData, firstdate::MIT)
    names = _ordered_colnames(sd.model)
    size(sd.values, 2) == length(names) ||
        error("MVTSeries(::SimData, ...): SimData column count " *
              "$(size(sd.values, 2)) != ordered name count $(length(names))")
    return MVTSeries(firstdate, names, sd.values)
end

# ----------------------------------------------------------------------
# (2) SimData(mvts::MVTSeries, model, maxlag, T) - adopt an existing
#     MVTSeries' backing matrix into a SimData. Column order must match
#     the model's vars-then-shocks declaration order; eltype must be
#     Float64. The SimData wraps the same backing storage.
# ----------------------------------------------------------------------

function Plans.SimData(mvts::MVTSeries, model::ModelBaseEcon.CompiledModel,
                       maxlag::Int, T::Int)
    expected = _ordered_colnames(model)
    got = collect(keys(getfield(mvts, :columns)))
    got == expected ||
        error("SimData(::MVTSeries, ...): column names mismatch. " *
              "Expected $(expected); got $(got).")
    vals = getfield(mvts, :values)
    eltype(vals) === Float64 ||
        error("SimData(::MVTSeries, ...): expected Float64 backing matrix, " *
              "got $(eltype(vals)).")
    size(vals, 1) == maxlag + T ||
        error("SimData(::MVTSeries, ...): row count $(size(vals, 1)) " *
              "!= maxlag + T = $(maxlag + T).")
    def = model.defs
    n_var = length(def.vars)
    n_shock = length(def.shocks)
    col_index = Dict{Symbol, Int}()
    is_log_col = falses(n_var + n_shock)
    for (i, v) in pairs(def.vars)
        col_index[v.name] = i
        is_log_col[i] = v.kind === ModelBaseEcon.IR.VAR_LOG
    end
    for (i, s) in pairs(def.shocks); col_index[s.name] = n_var + i;    end
    # maxlead = 0: the longbase/MVTSeries adoption path is the lead-free
    # workflow (SimData carries maxlag + T rows, no terminal rows).
    return SimData(model, vals, col_index, is_log_col, n_var, n_shock,
                   maxlag, T, 0)
end

# ----------------------------------------------------------------------
# (3) load_longbase_mvts - thin wrapper over Plans.load_longbase that
#     returns the resulting matrix as an MVTSeries with the model's
#     vars-then-shocks column labelling. The first date is derived from
#     the CSV's `first_quarter` argument (e.g. "2022Q4").
# ----------------------------------------------------------------------

function _parse_quarter_to_mit(s::AbstractString)
    # accepts "YYYYQq" forms, case-insensitive on the Q.
    m = match(r"^\s*(\d{4})Q([1-4])\s*$"i, s)
    m === nothing && error("load_longbase_mvts: cannot parse first_quarter " *
                           "$(repr(s)); expected \"YYYYQq\".")
    y = parse(Int, m.captures[1])
    q = parse(Int, m.captures[2])
    return MIT{Quarterly{3}}(y, q)
end

function Plans.load_longbase_mvts(path::AbstractString,
                                  model::ModelBaseEcon.CompiledModel;
                                  first_quarter::AbstractString,
                                  n_rows::Int)
    out, _row_labels = load_longbase(path, model;
                                     first_quarter = first_quarter,
                                     n_rows = n_rows)
    names = _ordered_colnames(model)
    return MVTSeries(_parse_quarter_to_mit(first_quarter), names, out)
end

end # module TimeSeriesEconExt
