##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

using Printf

"""
    printmatrix(mat [, Val(F), colnames])

Display a matrix in full while controlling the formatting of each value and
optionally showing the column names.
  * `Val(F)` - display each number in the given format `F`. The format is in the
    form of a decimal point number where the whole part indicates the total
    width and the fractional part indicates the number of digits printed after
    the decimal point. Default is `Val(12.7)`
  * `colnames` - a list of names to display in the first row. The names are
    displayed as given, possibly with padding to match the width given in the
    `Val` argument. If any names are longer than that, they will not be
    truncated and so the display will not be aligned properly. Sorry about that!

"""
printmatrix(mat::AbstractMatrix, args...) = printmatrix(stdout, mat, args...)
# printmatrix(io::IO, mat::AbstractMatrix, args...) = printmatrix(io, mat, args...)
@generated function printmatrix(io::IO, mat::AbstractMatrix, ::Val{F}=Val(12.7), ::Val{g}=Val(:g), cols=nothing, rows=nothing, ::Val{RW}=Val(20)) where {F,g,RW}
    N = floor(Int, F)
    digits = parse(Int, split("$(F-N)", ".")[2])
    fmts = "% $(N)s "
    fmtn = "% $F$g "
    row_fmt = "% -$(RW)s "
    return quote
        m, n = size(mat)
        if cols !== nothing
            if !isnothing(rows)
                Printf.@printf(io, $row_fmt, "")
            end
            for j ∈ 1:n
                c = string(cols[j])
                if length(c) > $N
                    c = c[1:$N-1] * "…"
                end
                Printf.@printf(io, $fmts, c)
            end
            println(io)
        end
        for i in 1:m
            if !isnothing(rows)
                c = string(rows[i])
                if length(c) > $RW
                    c = c[1:$RW-1] * "…"
                end
                Printf.@printf(io, $row_fmt, c)
            end
            for j in 1:n
                # s *= Printf.@sprintf($fmtn, mat[i, j])
                Printf.@printf(io, $fmtn, round(mat[i, j]; digits=$digits))
            end
            println(io)
        end
        return nothing
    end
end
export printmatrix

function ModelBaseEcon.transform(data::AbstractMatrix{Float64}, m::Model)
    tdata = copy(data)
    for (i, v) in enumerate(m.varshks)
        if need_transform(v)
            tdata[:, i] .= transform(data[:, i], v)
        end
    end
    return tdata
end

function ModelBaseEcon.inverse_transform(data::AbstractMatrix{Float64}, m::Model)
    idata = copy(data)
    for (i, v) in enumerate(m.varshks)
        if need_transform(v)
            idata[:, i] .= inverse_transform(data[:, i], v)
        end
    end
    return idata
end
