module StackedTimeSolver

using ModelBaseEcon
using ModelBaseEcon: IR, Symbolic
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC, sparse, nzrange, rowvals, nonzeros

export StackedTimePlan, stacked_RJ!, stacked_residual!, simulate!

include("stackedtime/solverdata.jl")
include("stackedtime/sparse.jl")
include("stackedtime/simulate.jl")

end # module StackedTimeSolver
