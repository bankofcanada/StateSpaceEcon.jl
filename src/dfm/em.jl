##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################


function EMstep(kfd::Kalman.AbstractKFData{RANGE,NS,NO,T}, Y, wks::DFMKalmanWks,
        Wlam = nothing, qlam = nothing
    ) where {RANGE,NS,NO,T}

    if Y isa MVTSeries
        Y = Y.values
    end

    ret = deepcopy(wks)
    for p in propertynames(ret)
        fill!(ret.:($p), 0)
    end

    nobs = RANGE isa UnitRange ? length(RANGE) : Int(RANGE)

    have_mu = !isnothing(wks.μ)

    # TMPxx = Matrix{Float64}(undef, NS, NS)
    # TMPxy = Matrix{Float64}(undef, NS, NO)

    XᵀX = ret.Txx
    YᵀX = ret.Λ

    mx = Vector{Float64}(undef, NS)

    sum!(XᵀX, kfd.Px_smooth)
    BLAS.gemm!('N', 'T', 1.0, kfd.x_smooth, kfd.x_smooth, 1.0, XᵀX)
    BLAS.gemm!('T', 'T', 1.0, Y, kfd.x_smooth, 0.0, YᵀX)

    if have_mu 
        sum!(mx, kfd.x_smooth)
        BLAS.ger!(-1.0/nobs, mx, mx, XᵀX)
        my = sum!(ret.μ, transpose(Y))
        BLAS.ger!(-1.0/nobs, my, mx, YᵀX)
    end
    
    iXᵀX = Symmetric(wks.Txx, :U)
    iXᵀX.data[:] = XᵀX
    rdiv!(YᵀX, cholesky!(iXᵀX))
    # copyto!(ret.Λ, YᵀX)   ### no need since YᵀX === ret.Λ
    LAPACK.potri!('U', iXᵀX.data)

    if !isnothing(Wlam) 
        # Given constraints on Λ: Wlam * vec(Λ) = qlam
        nc = size(Wlam, 1)
        nl = NO*NS
        # constraint residuals
        rcons = Vector{Float64}(undef, size(Wlam, 1))
        if isnothing(qlam)
            BLAS.gemv!('N', -1.0, Wlam, vec(ret.Λ), 0.0, rcons)
        else
            copy!(rcons, qlam)
            BLAS.gemv!('N', -1.0, Wlam, vec(ret.Λ), 1.0, rcons)
        end
        maximum(abs, rcons) < 100eps() && @goto done_lambda_constraints

        # The formula is 
        #    A = kron(iXᵀX, wks.R)
        #    B = A * transpose(W)
        #    C = W * B
        #    ϰ = inv(C) * rcons
        #    Λ += B * ϰ

        Bᵀ = zeros(nc, nl) # this will actually contain Bᵀ
        Bᵀ3 = reshape(Bᵀ, nc, NO, NS)  
        W3 = reshape(Wlam, nc, NO, NS)
        Tcy = Matrix{Float64}(undef, nc, NO)
        for i = 1:NS
            BLAS.gemm!('N', 'T', 1.0, wks.R, view(W3, :, :, i), 0.0, Tcy)
            for j = 1:NS
                BLAS.axpy!(iXᵀX[i,j], Tcy, view(Bᵀ3, :, :, j))
            end
        end

        C = Symmetric(Matrix{Float64}(undef, nc, nc), :U)
        BLAS.gemm!('N', 'T', 1.0, Wlam, Bᵀ, 0.0, C.data)
        ldiv!(cholesky!(C), rcons)  # rcons now contains kappa
        BLAS.gemv!('T', 1.0, Bᵀ, rcons, 1.0, vec(ret.Λ))

        @label done_lambda_constraints
        nothing
    end

    if have_mu
        BLAS.gemv!('N', -1.0/nobs, ret.Λ, mx, 1.0/nobs, ret.μ)
    end

    return ret

    # rA = 1:499
    # A = X[rA, :]'X[rA, :] + sum(kfdl.Px_smooth[:, :, t] for t = rA)
    # B = X[1:499, :]'X[2:500, :] + sum(kfdl.Pxx_smooth[:, :, t] for t = 1:499)
    # rC = 2:500
    # C = X[rC, :]'X[rC, :] + sum(kfdl.Px_smooth[:, :, t] for t = rC)
    # D = Y'X


end


