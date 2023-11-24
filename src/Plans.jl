##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

"""
    Plans

Module part of StateSpaceEcon. This module implements the `Plan` data structure,
which is used in simulations. The plan object contains information about the
range of the simulation and which variables and shocks are exogenous or
endogenous at each period of the range.

### Constructors
  * [`Plan(model, range)`](@ref Plan)

### Modify the plan
  * [`exogenize!`](@ref), [`endogenize!`](@ref) - make variables exogenous or
    endogenous
  * [`exog_endo!`](@ref), [`endo_exog!`](@ref) - swap exogenous and endogenous
    variables
  * [`autoexogenize!`](@ref) - exogenize and endogenize variables according to
    the list in the model

"""
module Plans

using TimeSeriesEcon
using ModelBaseEcon

# exported from this module that are also exported from StateSpaceEcon
export Plan,
    exogenize!, endogenize!, exog_endo!, endo_exog!, autoexogenize!

# exported from this module, but meant for internal use in StateSpaceEcon
export plansum


"""
    Plan{T <: MIT}

A data structure representing the simulation plan. It holds information about
the time range of the simulation and which variables/shocks are exogenous at
each period.
"""
struct Plan{T<:MIT} <: AbstractVector{Vector{Symbol}}
    range::AbstractUnitRange{T}
    varshks::NamedTuple
    exogenous::BitArray{2}
end

Plan(model::Model, r::Union{MIT,Int}) = Plan(model, r:r)

# account for default frequencies
Plan{MIT{Quarterly}}(args...) = Plan{MIT{Quarterly{3}}}(args...)
Plan{MIT{HalfYearly}}(args...) = Plan{MIT{HalfYearly{6}}}(args...)
Plan{MIT{Yearly}}(args...) = Plan{MIT{Yearly{12}}}(args...)
Plan{MIT{Weekly}}(args...) = Plan{MIT{Weekly{7}}}(args...)

"""
    Plan(model, range)

Create a default simulation plan for the given model over the given range. The
range of the plan is augmented to include periods before and after the given
range, over which initial and final conditions will be applied.

Instead of a range, one could also pass in a single moment in time
([`MIT`](@ref TimeSeriesEcon.MIT)) instance, in which case it is interpreted as
a range of length 1.

"""
function Plan(model::Model, range::AbstractUnitRange)
    if !(eltype(range) <: MIT)
        range = UnitRange{MIT{Unit}}(range)
    end
    range = (first(range)-model.maxlag):(last(range)+model.maxlead)
    local varshks = model.varshks
    local N = length(varshks)
    local names = tuple(Symbol.(varshks)...) # force conversion of Vector{ModelVariable} to NTuple{N,Symbol}
    p = Plan(range, NamedTuple{names}(1:N), falses(length(range), length(varshks)))
    for (ind, var) = enumerate(varshks)
        if isshock(var) || isexog(var)
            p.exogenous[:, ind] .= [true]
        end
    end
    return p
end

#######################################
# AbstractVector interface 

Base.size(p::Plan) = size(p.range)
Base.axes(p::Plan) = (p.range,)
Base.length(p::Plan) = length(p.range)
Base.IndexStyle(::Plan) = IndexLinear()
Base.similar(p::Plan) = Plan(p.range, p.varshks, similar(p.exogenous))
Base.copy(p::Plan) = Plan(p.range, p.varshks, copy(p.exogenous))
TimeSeriesEcon.rangeof(p::Plan) = p.range

function Base.copyto!(dest::Plan, rng::AbstractUnitRange, src::Plan; verbose=false)
    if isempty(rng)
        verbose && @warn "Nothing to copy - empty range"
        return dest
    end
    rng ⊈ rangeof(dest) && throw(BoundsError(dest, rng))
    rng ⊈ rangeof(src) && throw(BoundsError(src, rng))
    if verbose && rng != rangeof(dest)
        @warn "Ranges not updated in destination plan: $(join(_mits_to_ranges(setdiff(rangeof(dest), rng)), ", "))"
    end
    idx1 = _offset(dest, rng)
    jdx1 = _offset(src, rng)
    # if same columns, 
    if dest.varshks == src.varshks   # ===, not ==, so we compare names too
        jdx2 = idx2 = axes(dest.exogenous, 2)
        copyto!(dest.exogenous, idx1, idx2, src.exogenous, jdx1, jdx2)
        return dest
    end
    # we have to copy columns one at a time
    if verbose
        not_in_dest = setdiff(keys(src.varshks), keys(dest.varshks))
        if !isempty(not_in_dest)
            @warn "Ignored source plan variables (missing in destination plan): $(join(not_in_dest, ", "))"
        end
        not_in_src = setdiff(keys(dest.varshks), keys(src.varshks))
        if !isempty(not_in_dest)
            @warn "Variables not updated in destination plan (missing in source plan): $(join(not_in_src, ", "))"
        end
    end
    for (var, i2) in pairs(dest.varshks)
        j2 = get(src.varshks, var, -1)
        if j2 > 0
            copyto!(dest.exogenous, idx1, i2:i2, src.exogenous, jdx1, j2:j2)
        end
    end
    return dest
end

Base.copyto!(dest::Plan, rng::MIT, scr::Plan; verbose=false) = Base.copyto!(dest, rng:rng, scr; verbose)
Base.copyto!(dest::Plan, src::Plan; verbose=false) = Base.copyto!(dest, intersect(rangeof(dest), rangeof(src)), src; verbose)

@inline _offset(p::Plan{T}, idx::T) where {T<:MIT} = convert(Int, idx - first(p.range) + 1)
@inline _offset(p::Plan{T}, idx::AbstractUnitRange{T}) where {T<:MIT} =
    _offset(p, first(idx)):_offset(p, last(idx))

## Integer index is taken as is
@inline Base.getindex(p::Plan, idx::Int) = begin
    nms = collect(keys(p.varshks))
    nms[p.exogenous[idx, :]]
end
@inline Base.getindex(p::Plan, idx::AbstractUnitRange{Int}) = p[p.range[idx]]
@inline Base.getindex(p::Plan, idx::Int, ::Val{:inds}) = findall(p.exogenous[idx, :])

# Index of the frequency type returns the list of exogenous symbols
@inline Base.getindex(p::Plan, idx::MIT) = p[_offset(p, idx)]

@inline function Base.getindex(p::Plan, idx::MIT, ::Val{:inds})
    first(p.range) ≤ idx ≤ last(p.range) || throw(BoundsError(p, idx))
    return p[_offset(p, idx), Val(:inds)]
end

# A range returns the plan trimmed over that exact range.
Base.getindex(p::Plan{MIT{Unit}}, rng::AbstractUnitRange{Int}) = p[UnitRange{MIT{Unit}}(rng)]
@inline function Base.getindex(p::Plan, rng::AbstractUnitRange)
    rng.start < p.range.start && throw(BoundsError(p, rng.start))
    rng.stop > p.range.stop && throw(BoundsError(p, rng.stop))
    return Plan(rng, p.varshks, p.exogenous[_offset(p, rng), :])
end

# A range with a model returns a plan trimmed over that range and extended for initial and final conditions.
Base.getindex(p::Plan{MIT{Unit}}, rng::AbstractUnitRange{Int}, m::Model) = p[UnitRange{MIT{Unit}}(rng), m]
@inline function Base.getindex(p::Plan{T}, rng::AbstractUnitRange{T}, m::Model) where {T<:MIT}
    # rng = (rng.start-m.maxlag):(rng.stop+m.maxlead)
    # return p[rng]
    copyto!(Plan(m, rng), rng, p)
end

Base.setindex!(p::Plan, x, i...) = error("Cannot assign directly. Use `exogenize` and `endogenize` to alter plan.")

#######################################
# query the exo-end status of a variable

@inline Base.getindex(p::Plan, vars::Union{AbstractString,Symbol,ModelVariable}...) = Base.getindex(p, Symbol[vars...])
@inline Base.getindex(p::Plan, vars::AbstractVector{<:Union{AbstractString,ModelVariable}}) = Base.getindex(p, Symbol[vars...])
@inline Base.getindex(p::Plan{T}, vars::AbstractVector{Symbol}) where {T} = begin
    var_inds = Int[p.varshks[vars]...]
    Plan{T}(p.range, NamedTuple{(vars...,)}(eachindex(vars)), p.exogenous[:, var_inds])
end

@inline Base.getindex(p::Plan, rng::AbstractUnitRange, vars::Union{AbstractString,Symbol,ModelVariable}...) = Base.getindex(p, rng, Symbol[vars...])
@inline Base.getindex(p::Plan, rng::AbstractUnitRange, vars::AbstractVector{<:Union{AbstractString,ModelVariable}}) = Base.getindex(p, rng, Symbol[vars...])
@inline Base.getindex(p::Plan{T}, rng::AbstractUnitRange{T}, vars::AbstractVector{Symbol}) where {T} = begin
    rng.start < p.range.start && throw(BoundsError(p, rng.start))
    rng.stop > p.range.stop && throw(BoundsError(p, rng.stop))
    var_inds = Int[p.varshks[vars]...]
    Plan{T}(rng, NamedTuple{(vars...,)}(eachindex(vars)), p.exogenous[_offset(p, rng), var_inds])
end


#######################################
# Pretty printing

Base.summary(io::IO, p::Plan) = print(io, typeof(p), " with range ", p.range)

# Used in the show() implementation below
function collapsed_range(p::Plan{T}) where {T<:MIT}
    ret = Pair{Union{T,UnitRange{T}},Vector{Symbol}}[]
    i1 = first(p.range)
    i2 = i1
    val = p[i1]
    @inline make_key() = i1 == i2 ? i1 : i1:i2
    for i in p.range[2:end]
        val_i = p[i]
        if val_i == val
            i2 = i
        else
            push!(ret, make_key() => val)
            i1 = i2 = i
            val = val_i
        end
    end
    push!(ret, make_key() => val)
end

Base.show(io::IO, ::MIME"text/plain", p::Plan) = show(io, p)
function Base.show(io::IO, p::Plan)
    # 0) show summary before setting :compact
    summary(io, p)
    isempty(p) && return
    # print(io, ":")
    nrow, ncol = displaysize(io)
    limit = get(io, :limit, true)
    cp = collapsed_range(p)
    # find the longest string left of "=>" for padding
    maxl = maximum(length ∘ string ∘ first, cp)
    if limit
        dcol = ncol - maxl - 6
    else
        dcol = typemax(Int)
    end
    function print_exog(names)
        if isempty(names)
            print(io, "∅")
            return
        end
        lens = cumsum(map(x -> length("$x, "), names))
        show = lens .< dcol
        show[1] = true
        print(io, join(names[show], ", "))
        if !all(show)
            print(io, ", …")
        end
    end
    if !limit || length(cp) <= nrow - 5
        for (r, v) in cp
            print(io, "\n  ", lpad("$r", maxl, " "), " → ")
            print_exog(v)
        end
    else
        top = div(nrow - 5, 2)
        bot = length(cp) - nrow + 6 + top
        for (r, v) in cp[1:top]
            print(io, "\n  ", lpad("$r", maxl, " "), " → ")
            print_exog(v)
        end
        print(io, "\n   ⋮")
        for (r, v) in cp[bot:end]
            print(io, "\n  ", lpad("$r", maxl, " "), " → ")
            print_exog(v)
        end
    end
end

#######################################
# export and import Plan instances 

export exportplan, importplan

"""
    exportplan(plan; options)
    exportplan(file, plan; options)

Display the plan or save it in a text file.

### Options
 * `alphabetical=false` - set to `true` to sort the variables. By default
   variables will be listed in the same order as in the model.
 * `exog_mark="X"` - a short string (ideally 1 character) to mark exogenous
   values.
 * `endo_mark="-"` - a short string (ideally 1 character) to mark endogenous
   values.
 * `delim=" "` - delimiter. Use `","`` to make it a CSV file.

See also [`importplan`](@ref)
"""
function exportplan end

"""
    importplan(file)

Read the plan from a text file.

See also [`exportplan`](@ref)
"""
function importplan end

exportplan(p::Plan; kwargs...) = exportplan(Base.stdout, p; kwargs...)
@inline exportplan(file::AbstractString, p::Plan; kwargs...) = (
    open(file, "w") do f
        exportplan(f, p; kwargs...)
    end
)

# center padding
_cpad(x, n::Integer, args...) = _cpad(string(x), Int(n), args...)
function _cpad(x::AbstractString, n::Integer, args...)
    lx = length(x)
    return signed(n) < 2 + lx ? lpad(x, n, args...) :
           lpad(rpad(x, (signed(n) + lx) ÷ 2, args...), n, args...)
end

function exportplan(io::IO, p::Plan;
    alphabetical=false,  # whether to sort variables
    exog_mark="X", endo_mark="-",  # symbols used for each class
    delim=" ",  # set to "," to get a CSV file (with 3 skip rows and 1 header row)
    _name_delim=delim,  # padding after NAME column
    _range_delim=delim    # padding between range columns
)
    summary(io, p)
    println(io)
    println(io, "Range: ", p.range)
    println(io, "Variables: ", p.varshks)
    width1 = 2 + maximum(length, (sprint(print, v; context=io, sizehint=20) for v in keys(p.varshks)))
    width1 = max(width1, 2 + length("NAME"))
    ranges, _ = zip(collapsed_range(p)...)
    width2 = 1 .+ map(length, (sprint(print, rng; context=io, sizehint=15) for rng in ranges))
    width2 = max.(width2, 1 + maximum(length, (exog_mark, endo_mark)))
    tf_matr = p.exogenous[map(x -> _offset(p, first(x)), [ranges...]), :]
    println(io, "($(exog_mark)) = Exogenous, ($(endo_mark)) = Endogenous:")
    print(io, lpad("NAME", width1), _name_delim)
    tmp = (lpad(rng, w) for (rng, w) in zip(ranges, width2))
    println(io, join(tmp, _range_delim))
    varind = pairs(p.varshks)
    if alphabetical
        varind = sort([varind...], by=Base.Fix2(getindex, 1),
            lt=(l, r) -> isless(string(l), string(r)))
    end
    for (var, ind) in varind
        print(io, lpad(var, width1), _name_delim)
        tmp = (_cpad(b ? exog_mark : endo_mark, w) for (w, b) in zip(width2, tf_matr[:, ind]))
        println(io, join(tmp, _range_delim))
    end
end

@inline function importplan(fname::AbstractString; kwargs...)
    open(fname, "r") do f
        importplan(f; kwargs...)
    end
end

function importplan(io::IO)
    # parse line 1. Example: "Plan{MIT{Quarterly}} with range 2000Q1:2100Q1"
    line = readline(io)
    m = match(r"Plan\{MIT\{(\w+|\w+\{\d+\})\}\} with range (\w+(?:\{\d+\})?:\w+(?:\{\d+\})?)", line)
    if m === nothing
        error("expected Plan{MIT{Frequency}} at the start of line 1, got ", line, ".")
    end
    freq = parse_frequency(m.captures[1], 1)
    rng = parse_range(m.captures[2], false, 1)
    # parse line 2. Example: "Range: 2000Q1:2100Q1"  range must match line 1
    line = readline(io)
    m = match(r"Range: (\w+(?:\{\d+\})?:\w+(?:\{\d+\})?)", line)
    if m === nothing
        error("expected Range: at the start of line 2, got ", line, ".")
    end
    rng1 = parse_range(m.captures[1], false, 2)
    if rng1 != rng
        error("ranges on lines 1 and 2 do not match.")
    end
    # parse line 3. Example: Variables: (a = 1, b = 2, c = 3)
    line = readline(io)
    m = match(r"Variables: (\([\w\s=,]+\))", line)
    if m === nothing
        error("expected Variables: at the start of line 3, got ", line, ".")
    end
    nt = parse_namedtuple(m.captures[1], 3)
    if Set(nt) != Set(1:length(nt))
        error("Indexes of variables on line 3 are not valid.")
    end
    # sort nt by its values (variable indexes)
    nt = (; sort!(collect(pairs(nt)), by=last)...)
    # parse line 4.  Example "(X) = Exogenous, (-) = Endogenous
    line = readline(io)
    m = match(r"\((.+?)\) = Exogenous, \((.+?)\) = Endogenous:", line)
    if m === nothing
        error("unexpected line 4: ", line, ".")
    end
    exog_mark, endo_mark = m.captures
    # parse line 5. Example "  NAME,  2000Q1:2010Q1, 2010Q2, 2010Q2:2020Q4"
    line = readline(io) * " "
    m = match(r"\s*NAME(.\S*)\s+(\w+(?:\{\d+\})?(?::\w+(?:\{\d+\})?)?([^\w\{\}:]\S*)\s*.*)", line)
    if m === nothing
        error("unexpected line 5: ", line, ".")
    end
    _name_delim = m.captures[1]
    _range_delim = m.captures[3]
    p = Plan{MIT{freq}}(rng, nt, falses(length(rng), length(nt)))
    ranges_inds = Vector{Int}[]
    for str in split(m.captures[2], _range_delim; keepempty=false)
        r = parse_range(strip(str), true, 5)
        push!(ranges_inds, [_offset(p, r);])
    end
    # parse the rest of it
    pat = Regex("\\s+(\\w+)\\s*$(_name_delim)\\s*(.*)")
    for i = 1:length(nt)
        line = readline(io)
        m = match(pat, line)
        if m === nothing
            error("failed to parse line ", 5 + i, ": ", line)
        end
        var = Symbol(m.captures[1])
        var_ind = p.varshks[var]
        vals = split(m.captures[2], _range_delim; keepempty=false)
        for (rng_ind, val) in zip(ranges_inds, vals)
            val = strip(val)
            if val == exog_mark
                p.exogenous[rng_ind, var_ind] .= true
            elseif val == endo_mark
                nothing
            else
                error("unexpected $val on line ", 5 + i)
            end
        end
    end
    # @info " " freq rng nt ranges
    return p
end

function parse_range(str, allow_mit=true, line=nothing)
    e = Meta.parse(str)
    if occursin(":", str) || !allow_mit
        ans = Meta.isexpr(e, :call) && e.args[1] == :(:) ? eval(e) : nothing
        if !(ans isa UnitRange{<:MIT})
            error("expected range, got ", str, line === nothing ? "." : " on line $line.")
        end
    else
        ans = Meta.isexpr(e, :call) && e.args[1] == :(*) ? eval(e) : nothing
        if !(ans isa MIT)
            error("expected MIT, got ", str, line === nothing ? "." : " on line $line.")
        end
    end
    return ans
end

function parse_frequency(str, line=nothing)
    e = Meta.parse(str)
    ans = e isa Symbol || e isa Expr ? eval(e) : nothing
    if !(ans isa Type && ans <: Frequency)
        error("expected frequency, got ", str, line === nothing ? "." : " on line $line.")
    end
    return sanitize_frequency(ans)
end

function parse_namedtuple(str, line=nothing)
    e = Meta.parse(str)
    ans = Meta.isexpr(e, :tuple) && all(Base.Fix2(Meta.isexpr, :(=)), e.args) ? eval(e) : nothing
    if !(ans isa NamedTuple{NAMES,NTuple{N,Int}} where {NAMES,N})
        error("expected NamedTuple, got ", str, line === nothing ? "." : " on line $line.")
    end
    ans
end

#######################################
# The user interface to modify plan instances.

"""
    setplanvalue!(plan, value, vars, range)
    setplanvalue!(plan, value, vars, date)
    setplanvalue!(plan, value, vars, dates)

Modify the status of the given variable(s) on the given date(s). If `value` is
`true` then variables become exogenous, otherwise they become endogenous.

"""
function setplanvalue!(p::Plan{T}, val::Bool, vars::Array{Symbol,1}, date::AbstractUnitRange{T}) where {T<:MIT}
    firstindex(p) ≤ first(date) && last(date) ≤ lastindex(p) || throw(BoundsError(p, date))
    idx1 = _offset(p, date)
    for v in vars
        idx2 = get(p.varshks, v, 0)
        if idx2 == 0
            throw(ArgumentError("Unknown variable or shock name: $v"))
        end
        p.exogenous[idx1, idx2] .= val
    end
    return p
end
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::AbstractArray{Symbol,1}, date::AbstractUnitRange{Int}) = setplanvalue!(p, val, vars, UnitRange{MIT{Unit}}(date))
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::AbstractArray{Symbol,1}, date::Integer) = setplanvalue!(p, val, vars, date*U:date*U)
setplanvalue!(p::Plan{MIT{Unit}}, val::Bool, vars::AbstractArray{Symbol,1}, date::MIT{Unit}) = setplanvalue!(p, val, vars, date:date)
setplanvalue!(p::Plan{T}, val::Bool, vars::AbstractArray{Symbol,1}, date::T) where {T<:MIT} = setplanvalue!(p, val, vars, date:date)
setplanvalue!(p::Plan, val::Bool, vars::AbstractArray{Symbol,1}, date) = (foreach(d -> setplanvalue!(p, val, vars, d), date); p)


"""
    exogenize!(plan, vars, date)

Modify the given plan so that the given variables will be exogenous on the given
dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such. `date` can
be a moment in time (same type as the plan), or a range or an iterable or a
container.

"""
exogenize!(p::Plan, var, date) = setplanvalue!(p, true, Symbol[var,], date)
exogenize!(p::Plan, vars::AbstractVector, date) = setplanvalue!(p, true, Symbol[vars...], date)
exogenize!(p::Plan, vars::Tuple, date) = setplanvalue!(p, true, Symbol[vars...], date)


"""
    endogenize!(plan, vars, date)

Modify the given plan so that the given variables will be endogenous on the
given dates. `vars` can be a `Symbol` or a `String` or a `Vector` of such.
`date` can be a moment in time (same type as the plan), or a range or an
iterable or a container.

"""
endogenize!(p::Plan, var, date) = setplanvalue!(p, false, Symbol[var,], date)
endogenize!(p::Plan, vars::AbstractVector, date) = setplanvalue!(p, false, Symbol[vars...], date)
endogenize!(p::Plan, vars::Tuple, date) = setplanvalue!(p, false, Symbol[vars...], date)


"""
    exog_endo!(plan, exog_vars, endo_vars, date)

Modify the given plan so that the given variables listed in `exog_vars` will be
exogenous and the variables listed in `endo_vars` will be endogenous on the
given dates. `exog_vars` and `endo_vars` can each be a `Symbol` or a `String` or
a `Vector` of such. `date` can be a moment in time (same type as the plan), or a
range or an iterable or a container.

"""
function exog_endo!(p::Plan, exog, endo, date)
    setplanvalue!(p, true, Symbol[exog...], date)
    setplanvalue!(p, false, Symbol[endo...], date)
end
function exog_endo!(p::Plan, exog::Union{AbstractString,Symbol,ModelVariable}, endo::Union{AbstractString,Symbol,ModelVariable}, date)
    setplanvalue!(p, true, Symbol[exog], date)
    setplanvalue!(p, false, Symbol[endo], date)
end

"""
    endo_exog!(plan, endo_vars, exog_vars, date)

Modify the given plan so that the given variables listed in `exog_vars` will be
exogenous and the variables listed in `endo_vars` will be endogenous on the
given dates. `exog_vars` and `endo_vars` can each be a `Symbol` or a `String` or
a `Vector` of such. `date` can be a moment in time (same type as the plan), or a
range or an iterable or a container.

"""
@inline endo_exog!(p::Plan, endo, exog, date) = exog_endo!(p, exog, endo, date)

"""
autoexogenize!(plan, model, date)

Modify the given plan according to the "autoexogenize" protocol defined in the
given model. All variables in the autoexogenization list become endogenous and
their corresponding shocks become exogenous over the given date or range. `date`
can be a moment in time (same frequency as the given plan), a range, an
iterable, or a container.

"""
function autoexogenize!(p::Plan, m::Model, date)
    auto_vars = keys(m.autoexogenize)
    auto_shks = values(m.autoexogenize)
    exog_endo!(p, auto_vars, auto_shks, date)
end

#######################################
# The user interface to prepare data for simulation.

TimeSeriesEcon.frequencyof(p::Plan) = frequencyof(p.range)
TimeSeriesEcon.firstdate(p::Plan) = first(p.range)
TimeSeriesEcon.lastdate(p::Plan) = last(p.range)

#######################################
# The internal interface to simulations code.

"""
    plansum(model, plan)

Return the total number of exogenous variables in the simulation plan. Periods
over which initial and final conditions are imposed are not counted in this sum.

"""
plansum(m::Model, p::Plan) = sum(p.exogenous[(1+m.maxlag):(end-m.maxlead), :])

export setexog!
"""
    setexog!(plan, t, vinds)

Modify the plan at time t such that `vinds` are exogenous and the rest are
endogenous.

"""
function setexog!(p::Plan, tt::Int, vinds)
    p.exogenous[tt, :] .= false
    p.exogenous[tt, vinds] .= true
end
setexog!(p::Plan{T}, tt::T, vinds) where {T<:MIT} = setexog!(p, _offset(p, tt), vinds)

"""
    count_exog_points(p::Plan, rng, vars)

Count the number of exogenous points for the given plan over the given range.

Example:
```
count_exog_points(p, :, m.exogenous)
```

See also [`count_endo_points`](@ref)
"""
function count_exog_points end

"""
    count_endo_points(p::Plan, rng, vars)

Count the number of endogenous points for the given plan over the given range.

Example:
```
count_endo_points(p, :, m.shocks)
```

See also [`count_exog_points`](@ref)
"""
function count_endo_points end

count_exog_points(p::Plan, ::Colon, vars) = count_points(Val(:exog), p, :, [p.varshks[Symbol(v)] for v in vars])
count_exog_points(p::Plan, rng::Union{MIT,AbstractUnitRange{<:MIT}}, vars) = count_points(Val(:exog), p, _offset(p, rng), [p.varshks[Symbol(v)] for v in vars])
count_endo_points(p::Plan, ::Colon, vars) = count_points(Val(:endo), p, :, [p.varshks[Symbol(v)] for v in vars])
count_endo_points(p::Plan, rng::Union{MIT,AbstractUnitRange{<:MIT}}, vars) = count_points(Val(:endo), p, _offset(p, rng), [p.varshks[Symbol(v)] for v in vars])
count_points(::Val{:exog}, p::Plan, rng, vars) = sum(p.exogenous[rng, vars])
count_points(::Val{:endo}, p::Plan, rng, vars) = sum(.!p.exogenous[rng, vars])
export count_endo_points, count_exog_points

#############

"""
    compare_plans(left, right; options)
    compare_plans(file, left, right; options)
    compare_plans(Base.stdout, left, right; options)

Returns an MVTSeries with summary information comparing the two plans.
Each cell in the MVTSeries have an interger value corresponding to the following rubric:

* 0 -> endogenous in both plans.
* 1 -> exogenous in the left plan only.
* 2 -> exogenous in the right plan only.
* 3 -> exogenous in both plans.

The MVTSeries only includes the overlap between the plans; variables, shocks, and periods only
included in oe of the plans will not be part of the return MVTSeries.

A detailed comparison can also be printed to a file with the `outfile` or `io` arguments.

### Options
* `outfile=""` - save the detailed comparison to the target outfile path, if provided.
* `io=nothing` - output the detailed comparison to a particular iobuffer.
* `diff=false` - only populate detailed comparison with variables and shocks 
  which differ between the plans.
* `alphabetical=false` - set to `true` to sort the variables. By default
  variables will be listed in the same order as in the left plan.
* `exog_mark="X"` - a short string (ideally 1 character) to mark exogenous
  values.
* `endo_mark="~"` - a short string (ideally 1 character) to mark endogenous
  values.
* `missing_mark="."` - a short string (ideally 1 character) to display when a
  variable is missing from one of the plans.
* `delim=" "` - delimiter. Use `","`` to make it a CSV file.
* `pagelines=0` - Set to a positive integer to enable pagination. Number is
  interpreted as the number of lines to repeat the header line (the one with the
  ranges).
* `summary=false` - print summary on plan differences to stdout.
* `legend=false` - print legend for comparison matrix to stdout.
"""
function compare_plans end
export compare_plans

compare_plans(file::AbstractString, left::Plan, right::Plan; kwargs...) = compare_plans(left, right; outfile=file, kwargs...)
compare_plans(io::Union{IOBuffer, Base.TTY}, left::Plan, right::Plan; kwargs...) = compare_plans(left, right; io=io, kwargs...)

function compare_plans(left::Plan, right::Plan; outfile = "",
        io = nothing, 
        alphabetical=false,  # whether to sort variables
        pagelines=0,
        exog_mark="X", endo_mark="~", missing_mark=".", # symbols used for each class
        delim=" ",  # set to "," to get a CSV file (with 3 skip rows and 1 header row)
        _name_delim=delim,  # padding after NAME column
        _range_delim=delim,    # padding between range columns
        diff = false, # whether to print only variables/shocks which differ
        summary = false, # whether to print summary information to stdout
        legend = false # whether to print the legend for the matrix numbers
    ) #
    left_vars = keys(left.varshks)
    right_vars = keys(right.varshks)
    aligned_right = deepcopy(left)
    aligned_right.exogenous .= 0
    copyto!(aligned_right, right; verbose=false)
    left_matrix = Int64.(left.exogenous)
    right_matrix = Int64.(aligned_right.exogenous) .* 2
    combined_matrix = left_matrix .+ right_matrix
    combined_mvts = MVTSeries(first(rangeof(left)), collect(keys(left.varshks)), combined_matrix)
    combined_mvts = combined_mvts[intersect(rangeof(left),rangeof(right)), intersect(left_vars, right_vars)]

    

    if summary || diff
        differences = combined_mvts .% 3 .== 0
    end
    if summary
        if left.range == right.range
            println("Same range: ", left.range)
        else
            println("Range  left: ", left.range)
            println("Range right: ", right.range)
        end
        
        if Set(left_vars) == Set(right_vars)
            println("Same variables.")
        else
            println("Variables only in left plan: ", setdiff(left_vars, right_vars))
            println("Variables only in right plan: ", setdiff(right_vars, left_vars))
            println(length(intersect(left_vars, right_vars)), " common variables.")
        end
        for var in keys(combined_mvts)
            mits = rangeof(combined_mvts[var])[.!differences[var]]
            if length(mits) > 0
                ranges = _mits_to_ranges(mits)
                print(":$var differs between the plans for the range(s) $(ranges[1])")
                for rng in ranges[2:end]
                    print(", $rng")
                end
                print(".\n")
            end
        end
    end
    
    if outfile == "" && io === nothing
        @goto return_section
    end

    # print to file
    doclose = false
    if io === nothing
        io = open(outfile, "w")
        doclose = true
    end
    
    println(io)
    frequencyof(left.range) == frequencyof(right.range) || TimeSeriesEcon.mixed_freq_error(left.range, right.range)
    if left.range == right.range
        println(io, "Same range: ", left.range)
    else
        println(io, "Range  left: ", left.range)
        println(io, "Range right: ", right.range)
    end
    if Set(left_vars) == Set(right_vars)
        println(io, "Same variables.")
    else
        println(io, "Variables only in left plan: ", setdiff(left_vars, right_vars))
        println(io, "Variables only in right plan: ", setdiff(right_vars, left_vars))
        println(io, length(intersect(left_vars, right_vars)), " common variables.")
    end
    width1 = 2 + maximum(length, (sprint(print, v; context=io, sizehint=20) for v in union(left_vars, right_vars)))
    width1 = max(width1, 2 + length("NAME"))
    ranges = let
        foo, _ = zip(collapsed_range(left)..., collapsed_range(right)...)
        bar = sort!(unique!([first(left.range) - 1, first(right.range) - 1, last.(foo)...]))
        [bar[i-1]+1:bar[i] for i = 2:length(bar)]
    end
    width2 = 1 .+ map(length, (sprint(print, rng; context=io, sizehint=15) for rng in ranges))
    width2 = max.(width2, 6 + 2maximum(length, (exog_mark, endo_mark, missing_mark)))
    println(io, "($(exog_mark)) = Exogenous, ($(endo_mark)) = Endogenous, ($(missing_mark)) = Missing:")
    header = (_cpad(rng, w) for (rng, w) in zip(ranges, width2))
    println(io, lpad("NAME", width1), _name_delim, join(header, _range_delim))
    allvars = unique([left_vars..., right_vars...])
    if alphabetical
        sort!(allvars)
    end
    for (lno, var) in enumerate(allvars)
        if diff && var ∈ keys(combined_mvts) && all(differences[var])
            continue
        end
        print(io, lpad(var, width1), _name_delim)
        tmp = String[]
        for (rng, w) in zip(ranges, width2)
            #  compute left mark
            row = _offset(left, first(rng))
            col = var in left_vars ? left.varshks[var] : -1
            lmark = col < 0 || !checkbounds(Bool, left.exogenous, row, col) ? missing_mark : left.exogenous[row, col] ? exog_mark : endo_mark
            # compute right mark
            row = _offset(right, first(rng))
            col = var in right_vars ? right.varshks[var] : -1
            rmark = col < 0 || !checkbounds(Bool, right.exogenous, row, col) ? missing_mark : right.exogenous[row, col] ? exog_mark : endo_mark
            push!(tmp, _cpad("$lmark $rmark", w))
        end
        println(io, join(tmp, _range_delim))
        if pagelines > 0 && rem(lno, pagelines) == 0
            println(io, "\n", lpad("NAME", width1), _name_delim, join(header, _range_delim), "\n")
        end
    end

    doclose && close(io)

    @label return_section
    if legend
        println("Comparison legend:")
        println("* 0 -> endogenous in both plans.")
        println("* 1 -> exogenous in the left plan only.")
        println("* 2 -> exogenous in the right plan only.")
        println("* 3 -> exogenous in both plans.")
    end

    return combined_mvts
end

"""
    _mits_to_ranges(v::Vector{<:MIT})

Converts a vector of MITs to a vector of continuous MIT ranges.
"""
function _mits_to_ranges(v::Vector{<:MIT})
    ranges = Vector{UnitRange{MIT{frequencyof(v[1])}}}()
    current_mit = v[1]
    current_adj = 0
    for el in v[2:end]
        if el == current_mit + current_adj +1
            current_adj += 1
        else
            push!(ranges, current_mit:current_mit+current_adj)
            current_mit = el
            current_adj = 0
        end   
    end
    push!(ranges, current_mit:current_mit+current_adj)
    return ranges
end

end # module Plans

using .Plans
export Plan,
    exogenize!, endogenize!,
    exog_endo!, endo_exog!,
    autoexogenize!,
    exportplan, importplan, compare_plans,
    count_endo_points, count_exog_points


