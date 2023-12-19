##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

abstract type AbstractKFModel end

# Users of the algorithms provided in this module must 
# define their own type derived from AbstractKFModel and 
# implement the functions declared below.

# The model is most generally assumed to have the following 
# structure
#
# x - vector of latent state variables
# v - vector of state noise variables
# y - vector of observed variables
# n - vector of observation noise variables
#
#   y[t] = H(x[t], n[t])
#   x[t] = F(x[t-1], v[t])
#
# The user must provide implementations of the functions H and F
# that compute the expected value and covariance of the left-hand-side,
# given the expected value and covariance of the x in the right-hand-side.
# The covariances of the noises, as well as any exogenous data, must be 
# handled by the user's implementation.


"""
    kf_predict_x!(t, xₜ, Pxₜ, xₜ₋₁, Pxₜ₋₁, model::AbstractKFModel, user_data...)

Compute the expected value and covariance matrix of the state variables at t
given the mean and covariance of the state variables at t-1.

Users must implement a method of this function for their type of model.

When called, `xₜ₋₁` and `Pxₜ₋₁` are given and contain the state and its
covariance at t-1. The implementation must compute the expected state and its
covariance at t, and write them in-place into `xₜ` and `Pxₜ` respectively.

`t` contains the period and can be used in case the state transition depends on
exogenous data.
"""
function kf_predict_x!(t, xₜ, Pxₜ, xₜ₋₁, Pxₜ₋₁, model::AbstractKFModel, user_data...)
    # implement transition equations here, e.g.,
    #   xₜ[:] = F * xₜ₋₁
    #   Pxₜ[:,:] = F * Pxₜ₋₁ * F' + Gv * Pvₜ * Gv'
    error("Not implemented for $(typeof(model)).")
end


"""
    kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, model::AbstractKFModel, user_data...)

Compute the expected value and covariance matrix of the observed variables at t
given the mean and covariance of the state variables at t.

Users must implement a method of this function for their type of `model`.

When called, `xₜ` and `Pxₜ` are given and contain the state and its covariance
at t. The implementation must compute the expected observation vector, its
covariance, and the state-observed covariance at t, and write them in-place into
`yₜ`, `Pyₜ` and `Pxyₜ` respectively.

`t` contains the period and can be used in case the state transition depends on
exogenous data.
"""
function kf_predict_y!(t, yₜ, Pyₜ, Pxyₜ, xₜ, Pxₜ, model::AbstractKFModel, user_data...)
    # implement observation equations here, e.g.,
    #   yₜ[:] = H * xₜ
    #   Pyₜ[:,:] = H * Pxₜ * H' + Gn * Pnₜ * Gn'
    #   Pxyₜ[:,:] = Pxₜ * H'
    error("Not implemented for $(typeof(model)).")
end

"""
    kf_true_y!(t, y, model::AbstractKFModel, user_data...)

Write the observed values at time `t` into `y`.

Users must implement a method of this function for their type of `model`.
"""
function kf_true_y!(t, yₜ, model::AbstractKFModel, user_data...)
    # fill y with the observations at time t, e.g.,
    #    yₜ[:] = data[t, :]
    error("Not implemented for $(typeof(model))")
end

"""
    kf_length_x(model::AbstractKFModel, user_data...)

Return the number of state variables.

Users must implement a method of this function for their type of `model`.
"""
function kf_length_x(model::AbstractKFModel, user_data...)
    # return the length of the state vector x
    error("Not implemented for $(typeof(model))")
end

"""
    kf_length_y(model::AbstractKFModel, user_data...)

Return the number of observed variables.

Users must implement a method of this function for their type of `model`.
"""
function kf_length_y(model::AbstractKFModel, user_data...)
    # return the length of the observed vector y
    error("Not implemented for $(typeof(model))")
end

