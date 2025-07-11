##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################


"""
    kf = kf_filter(Y, x0, Px0, model, user_data...; options)

Main function call to run the Kalman Filter. It dispatches to a specific
    implementation that is best suited for the given inputs.

# Arguments

`Y` is the observed data. Rows (fist index) correspond to time, columns
    (second index) correspond to observed variables.

`x0` and `Px0` are the initial states and state covariance. Option
    `fwdstate` controls whether these are given at t = 0 (`fwdstate=true`,
    default) or at t = 0 (`fwdstate=false`)

`model, user_data...`

# Options:

`anymissing` - set to `true` if Y contains missing data. Missing values must be
set to `NaN`. Default is computed from the given `Y` as
`anymissing=any(isnan, Y)`.

`fwdstate` - default value is `true`, meaning that the given initial states are
at time t=1. In this case the filtering iteration starts with the observation
equation. If set to `false`, it is understood the initial states are given at
time t=0. In this case the filtering iteration starts with the state equation.

"""
function kf_filter end


function kf_filter(Y::AbstractMatrix, x0::AbstractVector, Px0::AbstractMatrix,
    model, user_data...; kwargs...)

    if !kf_is_linear(model, user_data...)
        error("Not implemented for non-linear models yet.")
    end

    # construct the data container
    kfd = KFDataFilter(axes(Y, 1), model, user_data...)
    # construct the KFilter instance
    kf = KFilter(kfd)
    if kf_length_y(kf) != size(Y, 2)
        error("Invalid data size - number of columns in the data does not match number of observed variables in the model.")
    end
    # call the default Kalman filter
    kf_filter!(kf, Y, x0, Px0, model, user_data...; kwargs...)
    return kfd

end

function kf_filter!(kf::KFilter, Y::AbstractMatrix,
    x0::AbstractVector, Px0::AbstractMatrix,
    LM::KFLinearModel;
    fwdstate::Bool=true, anymissing::Bool=any(isnan, Y), kwargs...)

    # call the default linear Kalman filter
    dk_filter!(kf, Y, LM.mu, LM.H, LM.F, LM.Q, LM.R, LM.G, x0, Px0, fwdstate, anymissing)
    return kf
end

function kf_filter!(kf::KFilter, Y::AbstractMatrix,
    x0::AbstractVector, Px0::AbstractMatrix,
    model, user_data...; kwargs...)

    if kf_is_linear(model, user_data...)
        return kf_filter!(kf, Y, x0, Px0, kf_linear_model(model, user_data...); kwargs...)
    else
        error("Not implemented for non-linear models yet.")
    end
end

function kf_smoother(Y::AbstractMatrix, x0::AbstractVector, Px0::AbstractMatrix,
    model, user_data...; kwargs...)

    if !kf_is_linear(model, user_data...)
        error("Not implemented for non-linear models yet.")
    end

    # construct the data container
    kfd = KFDataSmoother(axes(Y, 1), model, user_data...)
    # construct the KFilter instance
    kf = KFilter(kfd)
    if kf_length_y(kf) != size(Y, 2)
        error("Invalid data size - number of columns in the data does not match number of observed variables in the model.")
    end
    # Call the default Kalman filter and then smoother
    kf_filter!(kf, Y, x0, Px0, model, user_data...; kwargs...)
    kf_smoother!(kf, model, user_data...; kwargs...)
    return kfd

end

function kf_smoother!(kf::KFilter, LM::KFLinearModel; fwdstate::Bool=true, kwargs...)
    dk_smoother!(kf, LM.mu, LM.H, LM.F, LM.Q, LM.R, LM.G, fwdstate)
    return kf
end

function kf_smoother!(kf::KFilter, model, user_data...; kwargs...)
    if kf_is_linear(model, user_data...)
        return kf_smoother!(kf, kf_linear_model(model, user_data...); kwargs...)
    else
        error("Not implemented for non-linear models yet.")
    end
end

