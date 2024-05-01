##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

function EMinit!(wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, M::DFM, em_wks::EM_Wks{T}) where {T}

    # use this to overwrite only NaN values (preserving any initial guesses already given)
    assign_non_nan!(DEST, row_inds, col_inds, SRC) = begin
        for (sj, dj) in enumerate(col_inds)
            for (si, di) in enumerate(row_inds)
                if isnan(DEST[di, dj])
                    DEST[di, dj] = SRC[si, sj]
                end
            end
        end
    end

    @unpack model, params = M
    @unpack μ, Λ, R, A, Q = wks
    NT, NO = size(EY)

    resY = copy(EY)

    for obs_wks in em_wks.observed.loadings
        @unpack yinds, mean_estim = obs_wks
        μ_b = view(μ, yinds)
        if mean_estim
            mean!(μ_b, transpose(view(resY, :, yinds)))
        end
        if !iszero(μ_b)
            resY[:, yinds] .-= transpose(μ_b)
        end
    end

    # count the number of static factors in each components block
    nfvec = Vector{Int}(undef, length(model.components))
    for (i, c) in enumerate(values(model.components))
        # idiosyncratic don't count
        nfvec[i] = c isa IdiosyncraticComponents ? 0 : nstates(c)
    end
    # total number of static factors in the model
    NF = sum(nfvec)
    # TODO: use TSVD in the case of NO >> NF
    # Initial static factors (no lags) by PCA via SVD
    SY = svd(resY)
    FALL = view(SY.U, :, 1:NF)
    rmul!(FALL, Diagonal(view(SY.S, 1:NF)))

    # build dynamic factor (lags as needed) in kfd.x_smooth
    foffset = 0  # offset to keep track of which static factors belong to which block
    for (nf, tr_wks, (cn, cb)) in zip(nfvec, em_wks.transition.blocks, model.components)
        nf > 0 || continue # skip idiosyncratic
        xinds_1 = tr_wks.xinds_1
        @assert nf == length(xinds_1) == nstates(cb)
        NL = lags(cb)
        xinds = last(xinds_1) .+ (1-nf*NL:0)
        F = view(FALL, :, foffset .+ (1:nf))
        FT = view(kfd.x_smooth, xinds, NL:NT)
        for ind = 1:NL
            # lag = ind - 1
            FT[end-ind*nf.+(1:nf), :] .= transpose(F[begin+NL-ind:end-ind+1, :])
        end
        foffset += nf
    end

    for obs_wks in em_wks.observed.loadings
        @unpack XTX, YTX = obs_wks
        fill!(XTX, zero(T))
        fill!(YTX, zero(T))
    end

    # Estimate initial loadings
    foffset = 0  # offset to keep track of which static factors belong to which block
    for (nf, tr_wks, (cn, cb)) in zip(nfvec, em_wks.transition.blocks, model.components)
        nf > 0 || continue # skip idiosyncratic
        xinds_1 = tr_wks.xinds_1
        NL = lags(cb)
        if any(isnan, view(Λ, :, xinds_1))
            # compute initial loadings
            for (obs_wks, (on, ob)) in zip(em_wks.observed.loadings, model.observed)
                haskey(ob.components, cn) || continue # skip components that are not loaded by this observed block
                @unpack yinds, constraint, inds_cb, XTX, new_Λ = obs_wks
                any(isnan, view(Λ, yinds, xinds_1)) || continue # skip if loading is given

                ind = 1
                for name in keys(ob.components)
                    name == cn && break
                    ind = ind + 1
                end
                loc_ind = inds_cb[ind]

                NC = DFMModels.mf_ncoefs(ob)
                xinds_2 = last(xinds_1) .+ (1-nf*NC:0)

                @assert length(loc_ind) == nf * NC

                L = view(new_Λ, :, loc_ind)
                FTF = view(XTX, loc_ind, loc_ind)

                # L = Matrix{T}(undef, length(yinds), nf * NC)
                # FTF = Matrix{T}(undef, nf * NC, nf * NC)

                bY = view(resY, NL:NT, yinds)
                bFT = view(kfd.x_smooth, xinds_2, NL:NT)

                mul!(transpose(L), bFT, bY)
                mul!(FTF, bFT, transpose(bFT))
                cFTF = cholesky!(Symmetric(FTF, :U))
                rdiv!(L, cFTF)
            end
        end
        foffset += nf
    end

    # Apply loadings constraints
    for obs_wks in em_wks.observed.loadings
        @unpack yinds, xinds_estim, constraint, XTX, new_Λ = obs_wks
        cFTF = Cholesky(UpperTriangular(XTX))
        _apply_constraint!(new_Λ, constraint, cFTF, cov(view(resY, lags(model):NT, yinds)))
        assign_non_nan!(Λ, yinds, xinds_estim, new_Λ)
    end

    # Estimate transitions
    foffset = 0  # offset to keep track of which static factors belong to which block
    for (nf, tr_wks, (cn, cb)) in zip(nfvec, em_wks.transition.blocks, model.components)
        nf > 0 || continue # skip idiosyncratic
        xinds_1 = tr_wks.xinds_1
        NL = lags(cb)
        xinds = last(xinds_1) .+ (1-nf*NL:0)
        FT = view(kfd.x_smooth, xinds, NL:NT)

        F = view(SY.U, :, foffset .+ (1:nf))

        # Btw, also subtract contributions of factors
        # resY[NL:NT,:] -= transpose(FT) * transpose(view(Λ, :,xinds))
        mul!(transpose(view(resY, NL:NT, :)), view(Λ, :, xinds), FT, -1.0, 1.0)

        NCO = cb.order
        xinds_2 = last(xinds_1) .+ (1-nf*NCO:0)

        FMAT = Matrix{T}(undef, nf * NCO, nf * NCO)
        FRHS = Matrix{T}(undef, nf, nf * NCO)
        F_lag_ind(i) = 1+NCO-i:NT-i
        FRHS_lag_ind(i) = nf * (NCO - i) .+ (1:nf)
        for i = 1:NCO
            # FRHS = Fₜᵀ * [ Fₜ₋ᵢ for i = NCO, NCO-1, ..., 1 ]
            mul!(view(FRHS, :, FRHS_lag_ind(i)),
                transpose(view(F, F_lag_ind(0), :)),  # lag 0  of NCO
                view(F, F_lag_ind(i), :)  # lag i oc NCO
            )
            # NOTE: lags are placed backwards in the matrix
            #   That is, lag(i=1,j=1) is placed in the bottom right block 
            #   and so on until lag(i=NCO,j=NCO) is placed in the top left block. 
            #   In order to fill FMAT above the diagonal, we need only j <= i
            for j = 1:i
                mul!(view(FMAT, FRHS_lag_ind(i), FRHS_lag_ind(j)),
                    transpose(view(F, F_lag_ind(i), :)),   # lag i of NCO
                    view(F, F_lag_ind(j), :), # lag j of NCO
                )
            end
        end
        cFMAT = cholesky!(Symmetric(FMAT, :U))
        rdiv!(FRHS, cFMAT)
        _apply_constraint!(FRHS, tr_wks.constraint, cFMAT, cov(F))
        # assign A
        assign_non_nan!(A, xinds_1, xinds_2, FRHS)

        # update Q
        if any(isnan, view(Q, xinds_1, xinds_1))
            COV = zeros(nf, nf)
            V = Vector{T}(undef, nf)
            c = one(T) / (NT - NCO)
            for i in F_lag_ind(0)
                V[:] = view(F, i, :)
                for o in 1:NCO
                    mul!(V, view(FRHS, :, FRHS_lag_ind(o)), view(F, i - o, :), -1.0, 1.0)
                end
                BLAS.ger!(c, V, V, COV)
            end
            # assign Q
            assign_non_nan!(Q, xinds_1, xinds_1, COV)
        end
        foffset += nf
    end

    for (tr_wks, (cn, cb)) in zip(em_wks.transition.blocks, model.components)
        cb isa IdiosyncraticComponents || continue
        @unpack xinds_1, xinds_2 = tr_wks
        ns = nstates(cb)
        for (obs_wks, (on, ob)) in zip(em_wks.observed.loadings, model.observed)
            haskey(ob.components, cn) || continue
            NCO = cb.order
            FMAT = Matrix{T}(undef, NCO, NCO)
            FRHS = Matrix{T}(undef, 1, NCO)
            F_lag_ind(i) = 1+NCO-i:NT-i
            for (k, (yk, xk)) in enumerate(zip(obs_wks.yinds, xinds_1))
                for i = 1:NCO
                    FRHS[1, NCO-i+1] = dot(resY[F_lag_ind(0), yk], resY[F_lag_ind(i), yk])
                    for j = 1:i
                        FMAT[NCO-i+1, NCO-j+1] = dot(resY[F_lag_ind(i), yk], resY[F_lag_ind(j), yk])
                    end
                end
                rdiv!(FRHS, cholesky!(Symmetric(FMAT, :U)))
                for (l, xl) in enumerate(xinds_2[k:ns:end])
                    if isnan(A[xk, xl])
                        A[xk, xl] = FRHS[1, l]
                    end
                end
                if isnan(Q[xk, xk])
                    Q[xk, xk] = var(resY[i, yk] - sum(FRHS[end-j+1] * resY[i-j, yk] for j = 1:NCO) for i in F_lag_ind(0))
                end
            end
            break
        end
    end

    assign_non_nan!(R, axes(R)..., I(NO))
    return wks
end


