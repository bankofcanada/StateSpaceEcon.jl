##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

# This file contains a Kalman filter and smoother 
# implementation that is optimized 
# for stationary linear state space model 

# Implementation in this file follows 
# Time Series Analysis by State Space Methods, second edition,
# by J.Durbin and S.J.Koopman, 2012

# Notation in (Durbin & Koopman, 2012) is: 
#
#  yₜ   = Z αₜ + εₜ       εₜ ~ N(0, H)    
#      yₜ - vector varying length pₜ, 
#          set p = maximum(pₜ for t = 1:n)
#  αₜ₊₁ = T αₜ + R ηₜ     ηₜ ~ N(0, Q)
#      αₜ - vector fixed length m 
#   α₁ ~ N(a₁, P₁)
#
#   Yₙ - data for y -- matrix (n, p), NaN indicate missing observations
#
#   auₜ = E(αₜ | Yₜ)
#   Puₜ = Var(αₜ | Yₜ)
#   aₜ₊₁ = E(αₜ₊₁ | Yₜ)
#   Pₜ₊₁ = Var(αₜ₊₁ | Yₜ)

function dk_filter!(kf::KFilter, Y, mu, Z, T, H, Q, R, a₁, P₁,
    fwdstate::Bool=true,
    anymissing::Bool=any(isnan, Y)
)

    _Q = if R isa UniformScaling
        Q
    else
        R * Q * transpose(R)
    end

    have_mu = dot(mu, mu) > 0

    kfd = kf.kfd
    # implement the algorithm in 4.3.2 on p. 85

    K = Kₜ = kf.K

    x_pred = aₜ₊₁ = aₜ = kf.x_pred
    Px_pred = Pₜ₊₁ = Pₜ = kf.Px_pred

    x = auₜ = kf.x
    Px = Puₜ = kf.Px

    y_pred = kf.y_pred
    error_y = vₜ = kf.error_y
    Py_pred = Fₜ = kf.Py_pred

    ZP = similar(Z)
    TP = similar(Pₜ)

    if anymissing
        have_y = trues(kf.ny)
    end

    aₜ[:] = a₁
    Pₜ[:, :] = P₁

    if !fwdstate
        BLAS.gemv!('N', 1.0, T, a₁, 0.0, aₜ)
        BLAS.gemm!('N', 'N', 1.0, T, Pₜ, 0.0, TP)
        BLAS.gemm!('N', 'T', 1.0, TP, T, 0.0, Pₜ)
        BLAS.axpy!(1.0, _Q, Pₜ)
    end

    tstart, tstop = extrema(kf.range)
    all_y = true
    t = tstart
    while true

        # y_pred[:] = Z * aₜ
        BLAS.gemv!('N', 1.0, Z, aₜ, 0.0, y_pred)
        have_mu && BLAS.axpy!(1.0, mu, y_pred)

        # vₜ[:] = Y[t, :] - y_pred
        copyto!(vₜ, Y[t, :])
        BLAS.axpy!(-1.0, y_pred, vₜ)
        if anymissing
            map!(!isnan, have_y, vₜ)
            all_y = all(have_y)
        end

        if !all_y
            vₜ .*= have_y
        end

        # Either:  PZᵀ[:, :] = Pₜ * transpose(Z)
        # BLAS.gemm!('N', 'T', 1.0, Pₜ, Z, 0.0, PZᵀ)
        # Or: ZP[:, :] = Z * Pₜ
        # BLAS.symm!('R', 'U', 1.0, Pₜ, Z, 0.0, ZP)
        BLAS.gemm!('N', 'N', 1.0, Z, Pₜ, 0.0, ZP)

        # Fₜ[:, :] = Z * PZᵀ + H
        # BLAS.gemm!('N', 'N', 1.0, Z, PZᵀ, 1.0, Fₜ)
        BLAS.gemm!('N', 'T', 1.0, ZP, Z, 0.0, Fₜ)
        BLAS.axpy!(1.0, H, Fₜ)

        @kfd_set! kfd t y_pred Py_pred error_y

        ZTIF = @kfd_view kfd t aux_ZᵀPy⁻¹
        # Compute K = P Zᵀ / Fₜ using Cholesky factorization (faster than LU)
        if !all_y
            ny = sum(have_y)
            fill!(Kₜ, 0.0)
            fill!(ZTIF, 0.0)
            if ny == 0
                # if observations are missing, we cannot update the prediction
                # we set the Kalman gain, K, to zero, effectively 
                # giving auₜ = aₜ and Puₜ = Pₜ
                cFₜ = Cholesky(zeros(0,0), :U, 0)
                nothing
            else
                # delete rows of Z and rows and columns of F where observations are missing
                # and proceed with the same formulas but smaller matrices
                F⁺ = view(Fₜ, 1:ny, 1:ny)
                Z⁺ = view(Z, have_y, :)
                TMP = view(ZP, 1:ny, :)
                mul!(F⁺, mul!(TMP, Z⁺, Pₜ), transpose(Z⁺))
                F⁺[:,:] += view(H, have_y, have_y)
                cFₜ = cholesky!(Symmetric(F⁺))
                TMP[:,:] = Z⁺
                ldiv!(cFₜ, TMP)
                ZTIF[:, have_y] = transpose(TMP)
                mul!(view(Kₜ, :, have_y), Pₜ, transpose(TMP))
            end
        else
            # compute the Kalman gain K
            copyto!(Kₜ, transpose(ZP))
            cFₜ = cholesky!(Symmetric(Fₜ))
            rdiv!(Kₜ, cFₜ)
            # compute the auxiliary matrix Zᵀ⋅Fₜ⁻ꜝ
            copyto!(ZTIF, transpose(Z))
            rdiv!(ZTIF, cFₜ)
        end

        # auₜ[:] = aₜ + Kₜ * vₜ
        BLAS.gemv!('N', 1.0, Kₜ, vₜ, 0.0, auₜ)
        BLAS.axpy!(1.0, aₜ, auₜ)

        # Puₜ[:, :] = Pₜ - Kₜ * Z * Pₜ = (I-KₜZ)Pₜ
        # BLAS.gemm!('N', 'T', -1.0, Kₜ, PZᵀ, 1.0, Puₜ)
        # BLAS.gemm!('N', 'N', -1.0, Kₜ, ZP, 0.0, Puₜ)
        # BLAS.axpy!(1.0, Pₜ, Puₜ)
        mul!(copyto!(TP, I), Kₜ, Z, -1.0, 1.0)  # TP =  I - KₜZ
        mul!(Puₜ, TP, Pₜ) # Puₜ = TP * Pₜ

        @kfd_set! kfd t x_pred Px_pred K x Px

        # compute likelihood and residual-squared
        _assign_res2(kfd, t, error_y)
        if !all_y
            _assign_loglik(kfd, t, view(error_y, have_y), cFₜ)
        else
            _assign_loglik(kfd, t, error_y, cFₜ)
        end

        t = t + 1
        t > tstop && break

        # aₜ₊₁ .= T * (aₜ + Kₜ * vₜ)
        # aₜ₊₁[:] = T * auₜ
        BLAS.gemv!('N', 1.0, T, auₜ, 0.0, aₜ₊₁)

        # Pₜ₊₁ .= T * Pₜ * transpose(T) * transpose(I - Kₜ * Z) + _Q
        # Pₜ₊₁[:, :] = T * Puₜ * transpose(T) + _Q
        copyto!(Pₜ₊₁, _Q)
        BLAS.gemm!('N', 'N', 1.0, T, Puₜ, 0.0, TP)
        BLAS.gemm!('N', 'T', 1.0, TP, T, 1.0, Pₜ₊₁)

    end

    return
end



function dk_smoother!(kf::KFilter, Y, mu, Z, T, H, Q, R)

    #  note - in the book we have
    #     Kₜ = T Pₜ Zᵀ Fₜ⁻¹
    #     Lₜ = T - Kₜ Z
    #  However, in dk_filter! we calculate 
    #     Kₜ = Pₜ Zᵀ Fₜ⁻¹
    #  therefore for us
    #     Lₜ = T ( I - Kₜ Z )

    # Implement the algorithm in section 4.4.4 on p. 91
    #    rₜ₋₁ = Zᵀ Fₜ⁻¹ vₜ + Lₜᵀ rₜ
    #    Nₜ₋₁ = Zᵀ Fₜ⁻¹ Z + Lₜᵀ Nₜ Lₜ
    #    aˢₜ = aₜ + Pₜ rₜ₋₁
    #    Vₜ = Pₜ - Pₜ Nₜ₋₁ Pₜ
    # with initialization 
    #    rₙ = 0,  Nₙ = 0

    # From Table 4.4 on p. 104
    # Cov(aˢₜ, aˢⱼ) = Pₜ Lₜᵀ Lₜ₊₁ᵀ … Lⱼ₋₁ᵀ ( I - Nⱼ₋₁ Pⱼ) for j = t+1, ..., n
    # for j = t + 1, we have 
    #        Cov(aˢₜ, aˢₜ₊₁) = Pₜ Lₜᵀ ( I - Nₜ Pₜ₊₁ )

    have_mu = dot(mu, mu) > 0

    kfd = kf.kfd

    rₜ₋₁ = kf.x
    Nₜ₋₁ = kf.Px

    fill!(rₜ₋₁, 0)
    fill!(Nₜ₋₁, 0)

    Lₜ = Matrix{Float64}(undef, kf.nx, kf.nx)
    TMPx = Vector{Float64}(undef, kf.nx)
    TMPxx = Matrix{Float64}(undef, kf.nx, kf.nx)
    TMPxy = Matrix{Float64}(undef, kf.nx, kf.ny)

    tstart, tstop = extrema(kf.range)

    t = tstop + 1
    # Initialize Pₙ₊₁ with Pₙ -- close enough if n is large
    Pₜ = @kfd_view kfd tstop Px_pred
    #     to be exact: Pₜ = T * Pₙ * transpose(T) * transpose(I - Kₙ * Z) + _Q
    #     also, it gets multiplied by Nₙ, which is 0, so it doesn't matter!
    while t > tstart
        t = t - 1

        aₜ = @kfd_view kfd t x_pred
        Pₜ₊₁ = Pₜ
        Pₜ = @kfd_view kfd t Px_pred
        vₜ = @kfd_view kfd t error_y
        Kₜ = @kfd_view kfd t K
        ZᵀiFₜ = @kfd_view kfd t aux_ZᵀPy⁻¹

        # Lₜ[:,:] = T * ( I - Kₜ * Z )
        copyto!(TMPxx, I)
        BLAS.gemm!('N', 'N', -1.0, Kₜ, Z, 1.0, TMPxx)
        BLAS.gemm!('N', 'N', 1.0, T, TMPxx, 0.0, Lₜ)

        Pxx_smooth = @kfd_view kfd t Pxx_smooth
        # Pxx_smooth[:, :] = Pₜ * transpose(Lₜ) * (I - Nₜ * Pₜ₊₁)
        copyto!(Pxx_smooth, I)
        # NOTE: Nₜ₋₁ actually contains Nₜ, since it has not been updated this iteration yet
        t == tstop || BLAS.gemm!('N', 'N', -1.0, Nₜ₋₁, Pₜ₊₁, 1.0, Pxx_smooth)
        BLAS.gemm!('T', 'N', 1.0, Lₜ, Pxx_smooth, 0.0, TMPxx)
        BLAS.gemm!('N', 'N', 1.0, Pₜ, TMPxx, 0.0, Pxx_smooth)

        # rₜ₋₁[:] = (transpose(Z) / Fₜ) * vₜ + transpose(Lₜ) * rₜ
        copyto!(TMPx, rₜ₋₁)  # rₜ₋₁ and rₜ are stored in the same memory
        BLAS.gemv!('T', 1.0, Lₜ, TMPx, 0.0, rₜ₋₁)
        BLAS.gemv!('N', 1.0, ZᵀiFₜ, vₜ, 1.0, rₜ₋₁)

        # Nₜ₋₁[:,:] = (transpose(Z) / Fₜ) * Z + transpose(Lₜ) * Nₜ * Lₜ
        BLAS.gemm!('T', 'N', 1.0, Lₜ, Nₜ₋₁, 0.0, TMPxx)  # Nₜ and Nₜ₋₁ are stored in the same memory
        BLAS.gemm!('N', 'N', 1.0, TMPxx, Lₜ, 0.0, Nₜ₋₁)
        BLAS.gemm!('N', 'N', 1.0, ZᵀiFₜ, Z, 1.0, Nₜ₋₁)

        aˢₜ = @kfd_view kfd t x_smooth
        # aˢₜ[:] = aₜ + Pₜ * rₜ₋₁
        copyto!(aˢₜ, aₜ)
        BLAS.gemv!('N', 1.0, Pₜ, rₜ₋₁, 1.0, aˢₜ)

        Vₜ = @kfd_view kfd t Px_smooth
        # Vₜ[:,:] = Pₜ - Pₜ * Nₜ₋₁ * Pₜ
        copyto!(Vₜ, Pₜ)
        BLAS.gemm!('N', 'N', 1.0, Pₜ, Nₜ₋₁, 0.0, TMPxx)
        BLAS.gemm!('N', 'N', -1.0, TMPxx, Pₜ, 1.0, Vₜ)

        y_smooth = @kfd_view kfd t y_smooth
        # y_smooth[:] = mu + Z * x_smooth
        BLAS.gemv!('N', 1.0, Z, aˢₜ, 0.0, y_smooth)
        have_mu && BLAS.axpy!(1.0, mu, y_smooth)

        Py_smooth = @kfd_view kfd t Py_smooth
        # Py_smooth[:] = Z * Px_smooth * transpose(Z) + H
        copyto!(Py_smooth, H)
        BLAS.gemm!('N', 'T', 1.0, Vₜ, Z, 0.0, TMPxy)
        BLAS.gemm!('N', 'N', 1.0, Z, TMPxy, 1.0, Py_smooth)

    end

    return
end
