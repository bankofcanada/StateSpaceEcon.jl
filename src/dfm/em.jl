##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################


function EMstep(kfd::Kalman.AbstractKFData{RANGE,NS,NO,T}, Y, wks::DFMKalmanWks,
    cons::Union{DFMConstraints,Nothing}=nothing
) where {RANGE,NS,NO,T}

    if Y isa MVTSeries
        Y = Y.values
    end

    ret = deepcopy(wks)
    for p in propertynames(ret)
        fill!(ret.:($p), NaN)
    end

    nobs = RANGE isa UnitRange ? length(RANGE) : Int(RANGE)
    i_nobs = (one(T) / nobs)::T
    mi_nobs = -i_nobs

    have_mu = !isnothing(wks.μ)

    XᵀX = ret.Txx
    YᵀX = wks.Tyx

    @unpack μ, Λ, R = ret

    sx = Vector{Float64}(undef, NS)
    sy = Vector{Float64}(undef, NO)

    @unpack x_smooth, Px_smooth, Pxx_smooth = kfd

    sum!(XᵀX, Px_smooth)
    BLAS.gemm!('N', 'T', 1.0, x_smooth, x_smooth, 1.0, XᵀX)
    BLAS.gemm!('T', 'T', 1.0, Y, x_smooth, 0.0, YᵀX)


    if have_mu
        sum!(sx, x_smooth)
        BLAS.ger!(mi_nobs, sx, sx, XᵀX)
        sum!(sy, transpose(Y))
        BLAS.ger!(mi_nobs, sy, sx, YᵀX)
    end

    iXᵀX = Symmetric(wks.Txx, :U)
    iXᵀX.data[:] = XᵀX
    Λ[:] = YᵀX
    rdiv!(Λ, cholesky!(iXᵀX))
    # NOTE: the above, cholesky!(), computes the upper Cholesky factor
    #   and stores it in place in iXᵀX.data upper triangle

    if !isnothing(cons) && !isempty(cons.WΛ)
        @unpack WΛ, qΛ = cons
        # Given constraints on Λ: WΛ * vec(Λ) = qΛ

        nc = size(WΛ, 1)
        nl = NO * NS
        # constraint residuals
        rcons = Vector{Float64}(undef, nc)
        copyto!(rcons, qΛ)
        BLAS.gemv!('N', -1.0, WΛ, vec(Λ), 1.0, rcons)
        maximum(abs, rcons) < 100eps() && @goto done_Λ_constraints

        # @info "Applying Λ-constraints"

        # The formula is 
        #    A = kron(iXᵀX, wks.R)
        #    B = A * transpose(W)
        #    C = W * B
        #    ϰ = inv(C) * rcons
        #    Λ += B * ϰ

        # The upper Cholesky factor is in iXᵀX.data. LAPACK.potri! replaces it 
        # with the inverse
        LAPACK.potri!('U', iXᵀX.data)

        Bᵀ = zeros(nc, nl) # this will actually contain Bᵀ = W * Aᵀ (but A is symmetric, so there)
        Bᵀ3 = reshape(Bᵀ, nc, NO, NS)
        W3 = reshape(WΛ, nc, NO, NS)
        Tcy = Matrix{Float64}(undef, nc, NO)
        for i = 1:NS
            BLAS.gemm!('N', 'N', 1.0, view(W3, :, :, i), wks.R, 0.0, Tcy)
            for j = 1:NS
                BLAS.axpy!(iXᵀX[i, j], Tcy, view(Bᵀ3, :, :, j))
            end
        end

        C = Symmetric(Matrix{Float64}(undef, nc, nc), :U)
        BLAS.gemm!('N', 'T', 1.0, WΛ, Bᵀ, 0.0, C.data)
        ldiv!(cholesky!(C), rcons)  # rcons now contains kappa
        BLAS.gemv!('T', 1.0, Bᵀ, rcons, 1.0, vec(Λ))

        @label done_Λ_constraints
        nothing
    end

    if have_mu
        copyto!(μ, sy)
        BLAS.gemv!('N', mi_nobs, Λ, sx, i_nobs, μ)
    end

    # R = 1/nobs * ( YᵀY - YᵀXΛᵀ - ΛXᵀY + ΛXᵀXΛᵀ + Λ sum(Px_smooth) Λᵀ
    # In the case of have_mu = true, the YᵀX and XᵀX matrices have already been 
    #   corrected, but we still need to correct YᵀY.
    # Also, note that our XᵀX matrix already includes sum(Px_smooth)

    BLAS.gemm!('N', 'T', 1.0, YᵀX, Λ, 0.0, R)
    for i = 1:NO
        R[i, i] = 2 * R[i, i]
        for j = i+1:NO
            R[i, j] = R[j, i] = R[i, j] + R[j, i]
        end
    end
    BLAS.gemm!('T', 'N', 1.0, Y, Y, -1.0, R)
    BLAS.gemm!('N', 'N', 1.0, Λ, XᵀX, 0.0, ret.Tyx)
    BLAS.gemm!('N', 'T', i_nobs, ret.Tyx, Λ, i_nobs, R)

    if have_mu
        BLAS.ger!(mi_nobs * i_nobs, sy, sy, R)
    end

    # Now let's do A and Q

    PXpp = ret.Q
    PXmp = Matrix{Float64}(undef, NS, NS)
    PXmm = Matrix{Float64}(undef, NS, NS)

    @views begin
        sum!(PXmm, Px_smooth[:, :, begin:end-1])
        BLAS.gemm!('N', 'T', 1.0, x_smooth[:, begin:end-1], x_smooth[:, begin:end-1], 1.0, PXmm)
        sum!(PXpp, Px_smooth[:, :, begin+1:end])
        BLAS.gemm!('N', 'T', 1.0, x_smooth[:, begin+1:end], x_smooth[:, begin+1:end], 1.0, PXpp)
        sum!(PXmp, Pxx_smooth[:, :, begin:end-1])
        BLAS.gemm!('N', 'T', 1.0, x_smooth[:, begin:end-1], x_smooth[:, begin+1:end], 1.0, PXmp)
    end

    copyto!(ret.A, transpose(PXmp))

    iPXmm = Symmetric(copyto!(similar(PXmm), PXmm), :U)
    rdiv!(ret.A, cholesky!(iPXmm))

    if !isnothing(cons) && !isempty(cons.WA)
        @unpack WA, qA = cons

        nc = size(WA, 1)
        nl = NS * NS
        # constraint residuals
        rcons = Vector{Float64}(undef, nc)
        copyto!(rcons, qA)
        BLAS.gemv!('N', -1.0, WA, vec(ret.A), 1.0, rcons)
        maximum(abs, rcons) < 100eps() && @goto done_A_constraints

        # @info "Applying A-constraints"

        # The formula is 
        #    A = kron(iPXmm, wks.Q)
        #    B = A * transpose(W)
        #    C = W * B
        #    ϰ = inv(C) * rcons
        #    Λ += B * ϰ

        LAPACK.potri!('U', iPXmm.data)

        Bᵀ = zeros(nc, nl) # this will actually contain Bᵀ
        Bᵀ3 = reshape(Bᵀ, nc, NS, NS)
        W3 = reshape(WA, nc, NS, NS)
        Tcx = Matrix{Float64}(undef, nc, NS)
        for i = 1:NS
            BLAS.gemm!('N', 'T', 1.0, wks.Q, view(W3, :, :, i), 0.0, Tcx)
            for j = 1:NS
                BLAS.axpy!(iPXmm[i, j], Tcx, view(Bᵀ3, :, :, j))
            end
        end

        C = Symmetric(Matrix{Float64}(undef, nc, nc), :U)
        BLAS.gemm!('N', 'T', 1.0, WA, Bᵀ, 0.0, C.data)
        println(C)
        ldiv!(cholesky!(C), rcons)  # rcons now contains kappa
        BLAS.gemv!('T', 1.0, Bᵀ, rcons, 1.0, vec(ret.A))

        @label done_A_constraints
    end

    # copyto!(ret.Q, PXpp)    # no need, they occupy the same memory
    coef = one(T) / (nobs - 1)
    BLAS.gemm!('N', 'N', -coef, ret.A, PXmp, coef, ret.Q)

    return ret
end


