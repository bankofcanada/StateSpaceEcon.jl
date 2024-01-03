##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

# abstract type AbstractKFModel end

# Users of the algorithms provided in this module must 
# define their own model type and 
# implement the functions declared below.
# Make sure your methods specialize on your model type

# The model is most generally assumed to have the following 
# structure
#
# x - vector of latent state variables
# v - vector of state noise variables ~ N(0, Q)
# y - vector of observed variables
# w - vector of observation noise variables ~ N(0, R)
#
#   y[t]   = H(x[t], w[t])
#   x[t+1] = F(x[t], v[t])
#
# The user must provide implementations of the functions H and F
# that compute the expected value and covariance of the left-hand-side,
# given the expected value and covariance of the x in the right-hand-side.
# The covariances of the noises, as well as any exogenous data, must be 
# handled by the user's implementation.


"""
    kf_predict_x!(t, xₜ, Pxₜ, Pxxₜ₋₁ₜ, xₜ₋₁, Pxₜ₋₁, model, user_data...)

Compute the expected value and covariance matrix of the state variables at t
given the mean and covariance of the state variables at t-1.

Users must implement a method of this function for their type of model.

When called, `xₜ₋₁` and `Pxₜ₋₁` are given and contain the state and its
covariance at t-1. The implementation must compute the expected state, its
covariance at t, and the cross covariance of state at t-1 and t, 
and write them in-place into `xₜ`, `Pxₜ`, and `Pxxₜ₋₁ₜ` respectively.

Each of the output arguments can be `nothing`, which would indicate that they
don't need to be computed.

`t` contains the period and can be used in case the state transition depends on
exogenous data.
"""
function kf_predict_x!(t, xₜ, Pxₜ, Pxxₜ₋₁ₜ, xₜ₋₁, Pxₜ₋₁, model, user_data...)
    # implement transition equations here, e.g.,
    #   xₜ[:] = F * xₜ₋₁
    #   Pxₜ[:,:] = F * Pxₜ₋₁ * F' + Q * Pvₜ * Q'
    error("Not implemented for $(typeof(model)).")
end


"""
    kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, model, user_data...)

Compute the expected value and covariance matrix of the observed variables at t
given the mean and covariance of the state variables at t.

Users must implement a method of this function for their type of `model`.

When called, `xₜ` and `Pxₜ` are given and contain the state and its covariance
at t. The implementation must compute the expected observation vector, its
covariance, and the state-observed covariance at t, and write them in-place into
`yₜ`, `Pyₜ` and `Pxyₜ` respectively.

Each of the output arguments can be `nothing`, which would indicate that they
don't need to be computed.

`t` contains the period and can be used in case the state transition depends on
exogenous data.
"""
function kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, model, user_data...)
    # implement observation equations here, e.g.,
    #   yₜ[:] = H * xₜ
    #   Pyₜ[:,:] = H * Pxₜ * H' + R * Pwₜ * R'
    #   Pxyₜ[:,:] = Pxₜ * H'
    error("Not implemented for $(typeof(model)).")
end

"""
    kf_true_y!(t, y, model, user_data...)

Write the observed values at time `t` into `y`.

Users must implement a method of this function for their type of `model`.
"""
function kf_true_y!(t, yₜ, model, user_data...)
    # fill y with the observations at time t, e.g.,
    #    yₜ[:] = data[t, :]
    error("Not implemented for $(typeof(model))")
end

"""
    kf_length_x(model, user_data...)

Return the number of state variables.

Users must implement a method of this function for their type of `model`.
"""
function kf_length_x(model, user_data...)
    # return the length of the state vector x
    error("Not implemented for $(typeof(model))")
end

"""
    kf_length_y(model, user_data...)

Return the number of observed variables.

Users must implement a method of this function for their type of `model`.
"""
function kf_length_y(model, user_data...)
    # return the length of the observed vector y
    error("Not implemented for $(typeof(model))")
end

kf_islinear(model, user_data...) = false
kf_isstationary(model, user_data...) = false

function kf_linear_stationary(H, F, Q, R, model, user_data...)
    # fill the matrices H, F, Q, and R for the linear model
    #    y[t] = H * x[t] + v[t]      v ~ N(0, Q)
    #  x[t+1] = F * x[t] + w[t]      w ~ N(0, R)
    error("Not implemented for $(typeof(model))")
end


