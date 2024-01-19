##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

include("constraints.jl")

function EMstep!(wks::DFMKalmanWks, kfd::Kalman.AbstractKFData{RANGE,NS,NO,T}, Y,
    cΛ::Union{DFMConstraint,Nothing}=nothing,
    cA::Union{DFMConstraint,Nothing}=nothing
) where {RANGE,NS,NO,T}

    if Y isa MVTSeries
        Y = Y.values
    end

    #=
    ret = deepcopy(wks)
    for p in propertynames(ret)
        fill!(ret.:($p), NaN)
    end
    =#

    nobs = RANGE isa Number ? Int(RANGE) : length(RANGE)
    i_nobs = (one(T) / nobs)
    mi_nobs = -i_nobs

    have_mu = !isnothing(wks.μ)

    XᵀX = wks.Txx
    XᵀX_1 = wks.Txx_1
    YᵀX = wks.Tyx
    sx = wks.Tx
    sy = wks.Ty
    Tyx = wks.Tyx

    @unpack μ, Λ, R, A, Q = wks

    @unpack x_smooth, Px_smooth, Pxx_smooth = kfd

    # XᵀX = Xᵀ * X + sum( Pxx )
    sum!(XᵀX, Px_smooth)
    mul!(XᵀX, x_smooth, transpose(x_smooth), 1.0, 1.0)
    # YᵀX = Yᵀ * X = (Xᵀ * Y)ᵀ
    mul!(transpose(YᵀX), x_smooth, Y)

    if have_mu
        # correction for when mu is unknown and estimated
        sum!(sx, x_smooth)
        BLAS.ger!(mi_nobs, sx, sx, XᵀX)
        sum!(transpose(sy), Y)
        BLAS.ger!(mi_nobs, sy, sx, YᵀX)
    end

    # In-place Cholesky - overwrites XᵀX with the upper Cholesky factor
    copyto!(XᵀX_1, XᵀX)
    cXᵀX = cholesky!(Symmetric(XᵀX_1, :U))
    copyto!(Λ, YᵀX)
    rdiv!(Λ, cXᵀX)
    _apply_constraint!(Λ, cΛ, cXᵀX, wks.R)

    if have_mu
        copyto!(μ, sy)
        mul!(μ, Λ, sx, mi_nobs, i_nobs)
    end

    # R = 1/nobs * ( YᵀY - YᵀXΛᵀ - ΛXᵀY + ΛXᵀXΛᵀ + Λ sum(Px_smooth) Λᵀ )
    # In the case of have_mu = true, the YᵀX and XᵀX matrices have already been 
    #   corrected, but we still need to correct YᵀY.
    # Also, note that our XᵀX matrix already includes sum(Px_smooth)

    # start with R = YᵀXΛᵀ
    mul!(R, YᵀX, transpose(Λ))
    # double transpose to add ΛXᵀY
    for i = 1:NO
        R[i, i] = 2 * R[i, i]
        for j = i+1:NO
            R[i, j] = R[j, i] = R[i, j] + R[j, i]
        end
    end
    # negate and add YᵀY
    mul!(R, transpose(Y), Y, 1.0, -1.0)
    # add ΛXᵀXΛᵀ
    mul!(Tyx, Λ, XᵀX)
    mul!(R, Tyx, transpose(Λ), i_nobs, i_nobs)

    if have_mu
        BLAS.ger!(mi_nobs * i_nobs, sy, sy, R)
    end

    # Now let's do A and Q

    PXpp = Q       #  reuse memory
    PXmp = wks.Txx
    PXmm = wks.Txx_1
    TMP = wks.Txx_2

    begin
        mul!(PXmm, x_smooth, transpose(x_smooth))
        copyto!(PXpp, PXmm)
        v1 = @view x_smooth[:,begin]
        BLAS.ger!(-1.0, v1, v1, PXpp)
        v2 = @view x_smooth[:,end]
        BLAS.ger!(-1.0, v2, v2, PXmm)
        
        v3 = @view(x_smooth[:,begin:end-1])
        v4 = @view(x_smooth[:,begin+1:end])
        mul!(PXmp, v3, transpose(v4))

        sum!(PXmp, Pxx_smooth, init=false)
        BLAS.axpy!(-1.0, @view(Pxx_smooth[:,:,end]), PXmp)

        sum!(TMP, Px_smooth, init=true)
        BLAS.axpy!(1.0, TMP, PXpp)
        BLAS.axpy!(-1.0, @view(Px_smooth[:,:,begin]), PXpp)
        BLAS.axpy!(1.0, TMP, PXmm)
        BLAS.axpy!(-1.0, @view(Px_smooth[:,:,end]), PXmm)


        Amm = similar(PXmm)
        Amp = similar(PXmp)
        App = similar(PXpp)

        sum!(Amm, Px_smooth[:, :, begin:end-1])
        BLAS.gemm!('N', 'T', 1.0, x_smooth[:, begin:end-1], x_smooth[:, begin:end-1], 1.0, Amm)
        sum!(App, Px_smooth[:, :, begin+1:end])
        BLAS.gemm!('N', 'T', 1.0, x_smooth[:, begin+1:end], x_smooth[:, begin+1:end], 1.0, App)
        sum!(Amp, Pxx_smooth[:, :, begin:end-1])
        BLAS.gemm!('N', 'T', 1.0, x_smooth[:, begin:end-1], x_smooth[:, begin+1:end], 1.0, Amp)

        @assert Amm ≈ PXmm
        @assert Amp ≈ PXmp
        @assert App ≈ PXpp

    end

    copyto!(A, transpose(PXmp))
    copyto!(TMP, PXmm)
    cPXmm = cholesky!(Symmetric(TMP, :U))
    rdiv!(A, cPXmm)

    _apply_constraint!(A, cA, cPXmm, Q)

    # copyto!(Q, PXpp)    # no need, they occupy the same memory
    coef = one(T) / (nobs - 1)
    mul!(Q, A, PXmp, -coef, coef)

    return wks
end


