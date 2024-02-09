##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

include("constraints.jl")

function EMstep!(wks::DFMKalmanWks, kfd::Kalman.AbstractKFData{RANGE,NS,NO,T}, Y::AbstractMatrix,
    cΛ::Union{DFMConstraint,Nothing}=nothing,
    cA::Union{DFMConstraint,Nothing}=nothing,
    have_mu::Bool=true,
) where {RANGE,NS,NO,T}

    if Y isa MVTSeries
        Y = Y.values
    end

    nobs = RANGE isa Number ? Int(RANGE) : length(RANGE)
    i_nobs = (one(T) / nobs)
    mi_nobs = -i_nobs

    # have_mu = !isnothing(wks.μ)

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
    # cXᵀX = cholesky!(Symmetric(XᵀX_1, :U), check=false)
    # if cXᵀX.info != 0
    #     copyto!(XᵀX_1, XᵀX)
    #     XᵀX_1 += √eps(one(T)) * I(NS)
    #     cXᵀX = cholesky!(Symmetric(XᵀX_1, :U))
    # end
    copyto!(Λ, YᵀX)
    rdiv!(Λ, cXᵀX)
    _apply_constraint!(Λ, cΛ, cXᵀX, R)

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

    PXpp = wks.Txx
    PXmp = transpose(A)  # reuse memory
    PXmm = wks.Txx_1
    TMP = wks.Txx_2

    begin
        mul!(PXmm, x_smooth, transpose(x_smooth))
        copyto!(PXpp, PXmm)
        v1 = @view x_smooth[:, begin]
        BLAS.ger!(-1.0, v1, v1, PXpp)
        v2 = @view x_smooth[:, end]
        BLAS.ger!(-1.0, v2, v2, PXmm)

        v3 = @view(x_smooth[:, begin:end-1])
        v4 = @view(x_smooth[:, begin+1:end])
        mul!(PXmp, v3, transpose(v4))

        sum!(PXmp, Pxx_smooth, init=false)
        BLAS.axpy!(-1.0, @view(Pxx_smooth[:, :, end]), PXmp)

        sum!(TMP, Px_smooth, init=true)
        BLAS.axpy!(1.0, TMP, PXpp)
        BLAS.axpy!(-1.0, @view(Px_smooth[:, :, begin]), PXpp)
        BLAS.axpy!(1.0, TMP, PXmm)
        BLAS.axpy!(-1.0, @view(Px_smooth[:, :, end]), PXmm)


        #=      
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
        =#

    end

    # copyto!(A, transpose(PXmp))  # no need - they occupy the same memory
    copyto!(TMP, PXmm)
    cPXmm = cholesky!(Symmetric(TMP, :U))
    # cPXmm = cholesky!(Symmetric(TMP, :U), check=false)
    # if cPXmm.info != 0
    #     copyto!(TMP, PXmm)
    #     TMP += √eps(one(T)) * I(NS)
    #     cPXmm = cholesky!(Symmetric(TMP, :U))
    # end
    rdiv!(A, cPXmm)

    _apply_constraint!(A, cA, cPXmm, Q)

    copyto!(Q, PXpp)
    coef = one(T) / (nobs - 1)
    mul!(Q, A, PXmp, -coef, coef)

    return wks
end

using Printf

# nanmean(MAT) = nanmean!(similar(MAT, axes(MAT, 1)), MAT)
function nanmean!(mu, MAT)
    for i in eachindex(mu)
        s = 0.0
        c = 0
        for j in axes(MAT, 2)
            v = MAT[i, j]
            isnan(v) && continue
            s += v 
            c += 1
        end
        mu[i] = c == 0 ? 0 : (s / c)
    end
    return mu
end

function EMestimate(EM::DFM, Y::AbstractMatrix,
    wks::DFMKalmanWks=DFMKalmanWks(EM),
    x0::AbstractVector=zeros(Kalman.kf_length_x(EM)),
    Px0::AbstractMatrix=1e-10 * I(Kalman.kf_length_x(EM));
    maxiter=100, rftol=1e-4, axtol=1e-4,
    verbose=false,
    anymissing::Bool=any(isnan, Y)
)
    @unpack μ, Λ, A, R, Q = wks
    T = eltype(μ)

    conΛ = DFMConstraint(EM, Val(:Λ))
    conA = DFMConstraint(EM, Val(:A))

    EP = EM.params
    P = copy(EP)

    save_mu = similar(μ)

    if all(isnan, μ)
        have_mu = true  # means "estimate mu"
        if anymissing 
            nanmean!(save_mu, transpose(Y))
        else 
            mean!(save_mu, transpose(Y))
        end
        fill!(μ, zero(T))
    else
        have_mu = false  # means "mu is known to be zero"
        copyto!(save_mu, μ)
        fill!(μ, zero(T))
        DFMModels.set_mean!(P, EM.model, μ)
    end
    Y = Y .- transpose(save_mu)

    # initial guess
    # for i = eachindex(P)
    #     isnan(P[i]) || continue
    #     EP[i] = 0.1 + 0.05 * rand()
    # end
    # fill!(EP.observed.mean, zero(T))
    EMinit(EM, Y)
    _update_wks!(wks, EM)

    kfd = Kalman.KFDataSmoother(size(Y, 1), EM, Y, wks)
    kf = Kalman.KFilter(kfd)

    loglik = -Inf
    params = copy(EP)

    for iter = 1:maxiter

        Kalman.dk_filter!(kf, Y, μ, Λ, A, R, Q, I, x0, Px0, false, anymissing)
        Kalman.dk_smoother!(kf, Y, μ, Λ, A, R, Q, I)

        EMstep!(wks, kfd, Y, conΛ, conA, have_mu)

        copyto!(params, EP)
        _update_params!(EM, wks)
        for i = eachindex(P)
            isnan(P[i]) && continue
            EP[i] = P[i]
        end
        _update_wks!(wks, EM)

        loglik_new = sum(kfd.loglik)
        dx = maximum(abs, params - EP)
        df = 2 * abs(loglik - loglik_new) / (abs(loglik) + abs(loglik_new) + eps())

        if verbose == true && mod(iter, 10) == 0
            sl = @sprintf "%.6g" loglik
            sx = @sprintf "%.6g" dx
            sf = @sprintf "%.6g" df
            @info "EM iteration $(lpad(iter, 5)): loglik=$sl, df=$sf, dx=$sx"
        end

        if df < rftol
            break
        end

        loglik = loglik_new
    end

    Y .+= transpose(save_mu)
    DFMModels.get_mean!(μ, EM.model, EP)
    μ += save_mu
    DFMModels.set_mean!(EP, EM.model, μ)
    _update_wks!(wks, EM)
    Kalman.dk_filter!(kf, Y, μ, Λ, A, R, Q, I, x0, Px0, false)
    Kalman.dk_smoother!(kf, Y, μ, Λ, A, R, Q, I)

    return kf
end


function EMinit(EM::DFM, Y::AbstractMatrix)

    EP = EM.params
    m = EM.model

    for (bname, blk) in m.components
        p = getproperty(EP, bname)
        if blk isa IdiosyncraticComponents
            for i = 1:blk.size
                if isnan(p.covar[i])
                    p.covar[i] = 1
                end
                for l = 1:blk.order
                    if isnan(p.coefs[i, l])
                        p.coefs[i, l] = 0.1 * (l == 1)
                    end
                end
            end
        else
            for i = 1:blk.size
                for j = 1:blk.size
                    if isnan(p.covar[i, j])
                        p.covar[i, j] = i == j
                    end
                    for l = 1:blk.order
                        if isnan(p.coefs[i, j, l])
                            p.coefs[i, j, l] = 0.1 * (i == j && l == 1)
                        end
                    end
                end
            end
        end
    end

    for (oname, obs) in m.observed
        p = getproperty(EP, oname)
        fill!(p.mean, 0)
        for i = 1:nobserved(obs)
            if isnan(p.covar[i])
                p.covar[i] = 1
            end
        end
        for (bname, blk) in obs.components
            blk isa IdiosyncraticComponents && continue
            q = getproperty(p.loadings, bname)
            for i = 1:length(q)
                if isnan(q[i])
                    q[i] = 0.01
                end
            end
        end
    end

    return
end


