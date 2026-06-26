# ----------------------------------------------------------------------
# Kalman API.
#
# This file defines the API users must implement for their model type.
# ----------------------------------------------------------------------

"""
    kf_length_x(model, user_data...)

Return the number of state variables.

Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_length_x end

"""
    kf_length_y(model, user_data...)

Return the number of observed variables.

Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_length_y end

"""
    kf_is_linear(model, user_data...)

Return `true` if the `model` is a linear state space model.

Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_is_linear end

"""
    kf_linear_model(model, user_data...)

Create an instance of `KFLinearModel`, fill in the values of `mu`, `H`, `F`,
`G`, `Q`, `R` and return it.

Users must implement a method of this function for their type of `model` and `user_data`.
"""
function kf_linear_model end

"""
    kf_state_noise_shaping(model, user_data...)

Return `true` if the transition equation features a non-trivial noise shaping
    matrix, that is the matrix multiplying the shocks vector.

Return `false` if the state noise shaping matrix is the identity matrix.
"""
function kf_state_noise_shaping end
