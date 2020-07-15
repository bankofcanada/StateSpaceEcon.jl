
using Printf

printmatrix(mat, args...) = printmatrix(mat, Val(12.7), args...)
@generated function printmatrix(mat, ::Val{N}, cols=nothing) where N
    fmts = "% $(N)s "
    fmtn = "% $(N)f "
    return quote
        m, n = size(mat)
        if cols !== nothing
            s = ""
            for j ∈ 1:n
                s *= Printf.@sprintf($fmts, cols[j])
            end
            println(s)
        end
        for i in 1:m
            s = ""
            for j in 1:n
                s *= Printf.@sprintf($fmtn, mat[i,j])
            end
            println(s)
        end
        return nothing
    end
end
export printmatrix
