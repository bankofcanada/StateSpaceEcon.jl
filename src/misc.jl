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
    width and the fractional part is the number of digits printed after the
    decimal point. Default is `Val(12.7)`
  * `colnames` - a list of names to display in the first row. The names are
    displayed as given, possibly with padding to match the width given in the
    `Val` argument. If any names are longer than that, they will not be
    truncated and so the display will not be aligned properly. Sorry about that!

"""
@inline printmatrix(mat::AbstractMatrix, args...) = printmatrix(stdout, mat, args...)
@inline printmatrix(io::IO, mat::AbstractMatrix, args...) = printmatrix(io, mat, Val(12.7), args...)
@generated function printmatrix(io::IO, mat::AbstractMatrix, ::Val{N}, cols = nothing) where N
    fmts = "% $(N)s "
    fmtn = "% $(N)f "
    return quote
        m, n = size(mat)
        if cols !== nothing
            s = ""
            for j ∈ 1:n
                s *= Printf.@sprintf($fmts, cols[j])
            end
            println(io, s)
        end
        for i in 1:m
            s = ""
            for j in 1:n
                s *= Printf.@sprintf($fmtn, mat[i,j])
            end
            println(io, s)
        end
        return nothing
    end
end
export printmatrix

import ModelBaseEcon: transform, inverse_transform

function transform(data::AbstractMatrix{Float64}, m::Model) 
    tdata = similar(data)
    for (i, v) in enumerate(m.varshks)
        tdata[:, i] .= transform(data[:, i], v)
    end
    return tdata
end

function inverse_transform(data::AbstractMatrix{Float64}, m::Model) 
    idata = similar(data)
    for (i, v) in enumerate(m.varshks)
        idata[:, i] .= inverse_transform(data[:, i], v)
    end
    return idata
end
