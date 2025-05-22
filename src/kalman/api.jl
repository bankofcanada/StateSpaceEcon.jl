##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# This file defines the API users must implement for thier model type.

"""
    kf_length_x(model, user_data...)

Return the number of state variables.

Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_length_x end
# function kf_length_x(model, user_data...)
#     # return the length of the state vector x
#     error("Not implemented for $(typeof(model))")
# end

"""
    kf_length_y(model, user_data...)

Return the number of observed variables.

Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_length_y end
# function kf_length_y(model, user_data...)
#     # return the length of the observed vector y
#     error("Not implemented for $(typeof(model))")
# end

"""
    kf_is_linear(model, user_data...)

Return `true` if the `model` is a linear state space model.

Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_is_linear end
# function kf_is_linear(model, user_data...) 
#     error("Not implemented for $(typeof(model))")
# end

"""
    kf_linear_model(model, user_data...)
    
Create an instance of `KFLinearModel`, fill in the values of `mu`, `H`, `F`,
`G`, `Q`, `R` and return it.
    
Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_linear_model end
# kf_linear_model(model, user_data...) = error("Not implemented for $(typeof(model))")

"""
    kf_state_noise_shaping(model, user_data...)
    
Return `true` if the transition equation features a non-trivial noise shaping 
    matrix, that is the matrix multiplying the shocks vector.

Return `false` if the state noise shaping matrix is the identity matrix.
"""
function kf_state_noise_shaping end
# kf_state_noise_shaping(model, user_data...) = error("Not implemented for $(typeof(model))")


################################################################################


# """
#     kf_predict_x!(t, xₜ, Pxₜ, Pxxₜ₋₁ₜ, xₜ₋₁, Pxₜ₋₁, model, user_data...)

# Compute the expected value and covariance matrix of the state variables at t
# given the mean and covariance of the state variables at t-1.

# Users must implement a method of this function for their type of model.

# When called, `xₜ₋₁` and `Pxₜ₋₁` are given and contain the state and its
# covariance at t-1. The implementation must compute the expected state, its
# covariance at t, and the cross covariance of state at t-1 and t, 
# and write them in-place into `xₜ`, `Pxₜ`, and `Pxxₜ₋₁ₜ` respectively.

# Each of the output arguments can be `nothing`, which would indicate that they
# don't need to be computed.

# `t` contains the period and can be used in case the state transition depends on
# exogenous data.
# """
# function kf_predict_x!(t, xₜ, Pxₜ, Pxxₜ₋₁ₜ, xₜ₋₁, Pxₜ₋₁, model, user_data...)
#     # implement transition equations here, e.g.,
#     #   xₜ[:] = F * xₜ₋₁
#     #   Pxₜ[:,:] = F * Pxₜ₋₁ * F' + Q * Pvₜ * Q'
#     error("Not implemented for $(typeof(model)).")
# end


# """
#     kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, model, user_data...)

# Compute the expected value and covariance matrix of the observed variables at t
# given the mean and covariance of the state variables at t.

# Users must implement a method of this function for their type of `model`.

# When called, `xₜ` and `Pxₜ` are given and contain the state and its covariance
# at t. The implementation must compute the expected observation vector, its
# covariance, and the state-observed covariance at t, and write them in-place into
# `yₜ`, `Pyₜ` and `Pxyₜ` respectively.

# Each of the output arguments can be `nothing`, which would indicate that they
# don't need to be computed.

# `t` contains the period and can be used in case the state transition depends on
# exogenous data.
# """
# function kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, model, user_data...)
#     # implement observation equations here, e.g.,
#     #   yₜ[:] = H * xₜ
#     #   Pyₜ[:,:] = H * Pxₜ * H' + R * Pwₜ * R'
#     #   Pxyₜ[:,:] = Pxₜ * H'
#     error("Not implemented for $(typeof(model)).")
# end

# """
#     kf_true_y!(t, y, model, user_data...)

# Write the observed values at time `t` into `y`.

# Users must implement a method of this function for their type of `model`.
# """
# function kf_true_y!(t, yₜ, model, user_data...)
#     # fill y with the observations at time t, e.g.,
#     #    yₜ[:] = data[t, :]
#     error("Not implemented for $(typeof(model))")
# end


# function kf_linear_stationary(H, F, Q, R, model, user_data...)
#     # fill the matrices H, F, Q, and R for the linear model
#     #    y[t] = H * x[t] + v[t]      v ~ N(0, Q)
#     #  x[t+1] = F * x[t] + w[t]      w ~ N(0, R)
#     error("Not implemented for $(typeof(model))")
# end


