##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

# """
#     AbstractSolverData

# An abstract type for the internal data structure of a dynamic solver.

# ### Implementation (for developers)

# To implement a derived class, one must specialize the following methods. Note
# that in the signatures of most of these methods, the `sd` argument of the new
# type is the last argument..

#   + Constructor - normally it would take a model and a plan together with a
#     final condition type.
#   + `global_RJ(x, exog, sd)` - compute the residual and the Jacobian (return a
#     2-tuple) of the system at the given point `x` and exogenous data `exog`.
#   + `global_R!(res, x, exog, sd)` (optional) - compute the residual in place.
#   + `assign_exog_data!(x, exog, sd)` - this function applies the exogenous
#     constraints as per the plan and the given exogenous data.
#   + `assign_final_condition!(x, exog, sd, ::Val{T})` where T is one of the
#     FCType constants. This method applies the given type of final condition. If
#     T is fcgiven, the values of the final conditions are taken from exog.
#   + `assign_update_step(x, lambda, dx, sd)` - this method must implement x = x +
#     lambda * dx
#   + `update_plan!(sd, model, plan)` - update the solver data to reflect the given
#     simulation plan.

# """
# abstract type AbstractSolverData end

# """
#     global_RJ(x::Array, exog::Array, sd::AbstractSolverData)

# Compute the residual and the Jacobian of the dynamic system of equations at the
# given point `x` using the given exogenous data `exog`. Returns a tuple of two
# elements. The first is a vector of residuals and the second is the Jacobian
# matrix.

# """
# global_RJ(point::AbstractArray{Float64}, exog_data::AbstractArray{Float64}, g::AbstractSolverData) = error("Not implemented for $(typeof(g)).")
# # export global_RJ

# """
#     global_R!(res::Vector, x::Array, exog::Array, sd::AbstractSolverData)

# Compute the residual of the dynamic system of equations at the given point `x` using the given exogenous data `exog`.
# The residual is updated in place.
# """
# global_R!(res::AbstractArray{Float64}, point::AbstractArray{Float64}, exog_data::AbstractArray{Float64}, g::AbstractSolverData) = error("Not implemented for $(typeof(g)).")
# # export global_R!

# """
#     assign_exog_data(x::Array, exog::Array, sd::AbstractSolverData)

# Apply the exogenous constraints according to the simulation plan (passed to the
# constructor of sd) with exogenous values taken from `exog`.

# """
# assign_exog_data!(x::AbstractArray{Float64,2}, exog::AbstractArray{Float64,2}, g::AbstractSolverData) = error("Not implemented for $(typeof(g)).")
# # export assign_exog_data!

# """
#     assign_final_condition!(x::Array, exog::Array, sd::AbstractSolverData, ::Val{T})

# Apply final conditions of the given type. The type parameter `T` must be one of
# the constants of type `FCType`. If `T` is fcgiven, the final condition values
# are taken from `exog`. If `T` is one of `fclevel` or `fcslope`, then the final
# conditions are taken from the steady state solution in the model instance passed
# to the constructor of `sd`.

# """
# assign_final_condition!(x::AbstractArray{Float64,2}, exog::AbstractArray{Float64,2}, g::AbstractSolverData, ::Val{T}) where T = error("Not implemented for $(typeof(g)) with final condition $(T).")
# # export assign_final_condition!

# """
#     assign_update_step!(x::Array, lambda::Real, dx::Array, sd::AbstractSolverData)

# Assign the iterative update. Equivalent to `x = x + lambda * dx`.
# """
# function assign_update_step! end
# @inline assign_update_step!(x::AbstractArray{Float64}, lambda::Float64, Δx::AbstractArray{Float64}, ::AbstractSolverData) = BLAS.axpy!(lambda, Δx, x)  # axpy!(a,x,y) <=> y .= a .* x .+ y
# # export assign_update_step!

# """
#     update_plan!(sd, model, plan)

# Update the solver data structure for the given simulation plan. The `model` must
# be the same instance that was used to create `sd`. Typically, the plan should
# span the same range as the one used when `sd` was created, although it might
# have different exogenous/endogenous assignments.

# """
# update_plan!(sd::AbstractSolverData, m::Model, p::Plan) = error("Not implemented for $(typeof(sd)).")
# # export update_plan!
