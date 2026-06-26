# ----------------------------------------------------------------------
# Longbase CSV loader
# ----------------------------------------------------------------------
#
# The `longbase` file is a wide CSV: first column `OBS` holds a quarter
# label like "1962Q1", remaining columns are uppercase variable names.
# Missing entries are the string `NA`. This minimal loader pulls a
# requested quarter range into a plain `Matrix{Float64}` plus the row
# labels, matching column names to the model case-insensitively.

# Stub: the real method is provided by ext/TimeSeriesEconExt.jl when
# TimeSeriesEcon is loaded. Without the extension, calling this errors out
# with a clear message rather than the cryptic MethodError users get when
# a `Function` exists with no methods.
function load_longbase_mvts end

# Parse a "YYYYQq" label into an integer quarter ordinal (year*4 + q-1)
# so ranges are simple arithmetic.
function _quarter_ordinal(label::AbstractString)
    s = strip(label, ['"', ' '])
    m = match(r"^(\d{4})[Qq]([1-4])$", s)
    m === nothing && error("longbase: cannot parse quarter label $(repr(label))")
    year = parse(Int, m.captures[1])
    q = parse(Int, m.captures[2])
    return year * 4 + (q - 1)
end

_quarter_label(ord::Int) = string(ord ÷ 4, "Q", (ord % 4) + 1)

"""
    load_longbase(path, model; first_quarter, n_rows) -> (Matrix, row_labels)

Read the longbase CSV at `path` into a `(n_rows x n_col)` matrix whose
columns are the model's unified `[vars; shocks]` space, in *solver
space*: `@log` variable columns are stored as `log(level)` (the CSV
holds levels), matching the `@log` codegen convention. The matrix is
directly assignable into a `SimData.values`.

`first_quarter` is a label like `"2022Q1"`; `n_rows` consecutive
quarters are extracted. Columns present in the CSV but not in the model
are ignored; model columns absent from the CSV (e.g. the auto-shocks)
stay zero. `NA` entries are left at zero. Returns the matrix and the
quarter labels of the extracted rows.
"""
function load_longbase(path::AbstractString, model::ModelBaseEcon.CompiledModel;
                       first_quarter::AbstractString, n_rows::Int)
    isfile(path) || error("load_longbase: file not found: $path")
    def = model.defs
    n_var = length(def.vars)
    n_shock = length(def.shocks)
    col_index = Dict{Symbol, Int}()
    is_log_col = falses(n_var + n_shock)
    for (i, v) in pairs(def.vars)
        col_index[v.name] = i
        is_log_col[i] = v.kind === IR.VAR_LOG
    end
    for (i, s) in pairs(def.shocks); col_index[s.name] = n_var + i;    end

    lines = readlines(path)
    length(lines) >= 2 || error("load_longbase: file has no data rows")

    # Header: split on commas, strip quotes, lowercase -> Symbol.
    raw_headers = split(lines[1], ',')
    # raw_headers[1] is OBS; the rest are column names.
    csv_cols = [Symbol(lowercase(strip(h, ['"', ' ']))) for h in raw_headers]
    # Map CSV column position -> model unified column (0 if unmapped).
    csv_to_model = zeros(Int, length(csv_cols))
    for (j, name) in pairs(csv_cols)
        j == 1 && continue                       # OBS column
        csv_to_model[j] = get(col_index, name, 0)
    end

    start_ord = _quarter_ordinal(first_quarter)
    out = zeros(n_rows, n_var + n_shock)
    row_labels = String[_quarter_label(start_ord + k) for k in 0:n_rows-1]
    found = falses(n_rows)

    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ',')
        ord = _quarter_ordinal(fields[1])
        k = ord - start_ord
        (0 <= k < n_rows) || continue
        found[k + 1] = true
        for j in 2:min(length(fields), length(csv_cols))
            mc = csv_to_model[j]
            mc == 0 && continue
            v = strip(fields[j], ['"', ' '])
            (v == "NA" || isempty(v)) && continue
            level = parse(Float64, v)
            # @log columns store log(level): the CSV holds levels but the
            # solver works in log space for @log variables.
            out[k + 1, mc] = is_log_col[mc] ? log(level) : level
        end
    end
    all(found) || error("load_longbase: requested range $(first_quarter)" *
                        " + $(n_rows)q not fully present in $path")
    return out, row_labels
end
