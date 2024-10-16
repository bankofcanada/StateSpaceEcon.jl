##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

export kalman_filter!

function kalman_filter!(
    kd::KalmanData{NO,NS,NP},
    model, observed_data::AbstractArray{<:Real,2},
    initial_state::AbstractVector{<:Real},
    initial_state_covariance::AbstractMatrix{<:Real},
    ::Val{:plain}=Val(:plain)    #= method =#
) where {NO,NS,NP}

    @assert nobserved(model) == NO
    @assert nstates(model) == NS
    @assert size(observed_data) == (NP, NO)
    @assert size(initial_state) == (NS,)
    @assert size(initial_state_covariance) == (NS, NS)
    NSS = nstateshocks(model)
    NOS = nobservedshocks(model)

    # mean state shock 
    vbar = zeros(NSS)
    # mean observed shocks
    nbar = zeros(NOS)

    # period 0 - initial data
    x = Vector{Float64}(undef, NS)
    copyto!(x, initial_state)
    Px = Symmetric(Matrix{Float64}(undef, NS, NS))
    copyto!(Px.data, initial_state_covariance)

    # loop over the periods
    for period = 1:NP

        #########################################
        # prediction step (i.e., time update)

        # predicted state
        x_, F, Gv = eval_transition(model, x, vbar, period)
        kdstore!(kd, period; xpred=x_)

        # covariance of predicted state
        Rv = stateshocks_covariance(model, period)
        Px_ = Symmetric(F * Px * F' + Gv * Rv * Gv')
        kdstore!(kd, period; Pxpred=Px_)

        #########################################
        # correction step (i.e., measurement update)

        # predicted observation
        y_, H, Gn = eval_observation(model, x_, nbar, period)
        Rn = observedshocks_covariance(model, period)
        kdstore!(kd, period; ypred=y_)

        # observation error
        error_y = observed_data[period, :] - y_

        # predicted state-observed cross-correlation
        Pxy_ = Px_ * H'
        kdstore!(kd, period; Pxypred=Pxy_)

        # covariance of predicted observation
        Py_ = Symmetric(H * Pxy_ + Gn * Rn * Gn')
        kdstore!(kd, period; Pypred=Py_)
        Sy_ = cholesky(Py_)

        # kalman gain
        K = Pxy_ / Sy_
        kdstore!(kd, period; K)

        # updated state
        x[:] = x_ + K * error_y
        kdstore!(kd, period; x)

        # covariance of updated state 
        Px.data[:, :] = Px_ - K * H * Px_
        kdstore!(kd, period; Px)

        #########################################
        # conditional log likelihood
        sqr_res = error_y'error_y
        kdstore!(kd, period; sqr_res)
        logdet_Py_ = 2sum(log, diag(Sy_.U))
        scaled_error_y = ldiv!(Sy_.L, error_y) #! overwrites error_y
        log_lik = -0.5 * (NO * log(2π) + logdet_Py_ + scaled_error_y'scaled_error_y)
        kdstore!(kd, period; log_lik)
    end

    return kd

end


function kalman_smoother!(
    kd::KalmanData{NO,NS,NP},
    model,
    ::Val{:plain}=Val(:plain)    #= method =#
) where {NO,NS,NP}

    NSS = nstateshocks(model)
    NOS = nobservedshocks(model)

    # mean state shock 
    vbar = zeros(NSS)
    # mean observed shocks
    nbar = zeros(NOS)

    T = lastindex(kd.x, 1)
    xsmooth = kd.x[T, :]
    Pxsmooth = kd.Px[T, :, :]
    ysmooth, _, _ = eval_observation(model, xsmooth, nbar, T)
    kdstore!(kd, T; xsmooth, Pxsmooth, ysmooth)
    for period = reverse(1:T-1)
        _, F, _ = eval_transition(model, xsmooth, vbar, period)
        Px = kd.Px[period, :, :]
        Pxpred = kd.Pxpred[period+1, :, :]
        J = Px * F' / Pxpred

        xsmooth .= kd.x[period, :] + J * (xsmooth - kd.xpred[period+1, :])
        Pxsmooth .= Px + J * (Pxsmooth - Pxpred) * J'
        ysmooth, H, Gn = eval_observation(model, xsmooth, nbar, period)
        Rn = observedshocks_covariance(model, period)
        Pysmooth = H * Pxsmooth * H' + Gn * Rn * Gn'

        kdstore!(kd, period; xsmooth, Pxsmooth, ysmooth, Pysmooth, J)
    end

    return kd
end


