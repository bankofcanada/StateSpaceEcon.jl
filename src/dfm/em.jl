##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

function em_impute_observed!(EY::AbstractMatrix, kfd::Kalman.AbstractKFData, Y::AbstractMatrix)
    EY === Y && return
    @assert EY[.!isnan.(Y)] == Y[.!isnan.(Y)]
    YS = kfd.y_smooth  # this one is transposed (NO × NT)
    for i = axes(Y, 1)
        for j = axes(Y, 2)
            @inbounds yij = Y[i, j]
            yij == yij && continue  # EY is a copy of Y. The non-NaN values never change
            @inbounds EY[i, j] = YS[j, i] # update NaN value
        end
    end
    return EY
end



##################################################################################



struct EM_Observed_Block_Loading_Wks{T}
    yinds::Vector{Int}
    # xinds::Vector{Int}
    xinds_estim::Vector{Int}
    xinds_given::Vector{Int}
    mean_estim::Bool
    constraint::Union{Nothing,DFMConstraint{T}}
    # pre-allocated matrices
    YTX::Matrix{T}
    new_Λ::Matrix{T}
    XTX::Matrix{T}
    XTX_ge::Matrix{T}
    SY::Vector{T}
    SX::Vector{T}
end


function em_observed_block_loading_wks(on::Symbol, M::DFM, wks::DFMKalmanWks{T}) where {T}
    @unpack model, params = M
    @unpack μ, Λ, R = wks

    observed_m = observed(model)
    states_m = states_with_lags(model)
    ob = model.observed[on]
    yinds = append!(Int[], indexin(observed(ob), observed_m))

    if any(isnan, μ)
        mean_estim = true
        if !all(isnan, μ)
            @error("Estimating some, but not all, observed means is not supported. Will estimate all.")
        end
    else
        mean_estim = false
    end

    NC = DFMModels.mf_ncoefs(ob)

    states_b = mapfoldl(states_with_lags, append!, ob.components, init=Symbol[])
    xinds = append!(Int[], indexin(states_b, states_m))
    xinds_estim = Int[]
    xinds_given = Int[]
    offset = 0
    W = spzeros(0, 0)
    q = spzeros(0)
    for (cn, cb) in ob.components
        bcols = offset .+ (1:nstates_with_lags(cb))
        if any(isnan, view(Λ, :, xinds[bcols]))
            NS = nstates(cb)
            # we take all columns of the components block;
            # we assume lags(cb) ≥ NC (check is done before calling us); 
            # we take NC lags 
            xinds_cb_estim = xinds[bcols[end-NS*NC+1:end]]
            Wb, qb = DFMModels.loadingcons(view(Λ, yinds, xinds_cb_estim), ob, cb)
            W = blockdiag(W, Wb)
            q = vcat(q, qb)
            append!(xinds_estim, xinds_cb_estim)
            if length(bcols) > NS * NC
                # lags between NC and lags(cb) always have their loadings = 0
                xinds_cb_given = xinds[bcols[begin:end-NS*NC]]
                @assert iszero(view(Λ, yinds, xinds_cb_given))
                # append!(xinds_given, xinds_cb_given)
            end
        else
            # loadings of this block are given. 
            # Record the column indices of columns that are not all zero
            for c in bcols
                if !iszero(view(Λ, yinds, xinds[c]))
                    push!(xinds_given, xinds[c])
                end
            end
        end
        offset = last(bcols)
    end
    nest = length(xinds_estim)
    ngiv = length(xinds_given)
    nobs = length(yinds)
    return EM_Observed_Block_Loading_Wks{T}(
        yinds,
        # xinds, 
        xinds_estim, xinds_given,
        mean_estim,
        DFMConstraint(nest, W, q),
        Matrix{T}(undef, nobs, nest),     # YTX
        Matrix{T}(undef, nobs, nest),     # new_Λ
        Matrix{T}(undef, nest, nest),     # XTX
        Matrix{T}(undef, ngiv, nest),     # XTX_ge
        Vector{T}(undef, nobs),           # SY
        Vector{T}(undef, nest),           # SX
    )
end

function em_update_observed_block_loading!(wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, em_wks::EM_Observed_Block_Loading_Wks) where {T}
    # unpack the inputs
    @unpack yinds, xinds_estim, xinds_given, constraint = em_wks
    @unpack mean_estim = em_wks
    @unpack new_Λ, YTX, XTX, XTX_ge, SX, SY = em_wks
    @unpack x_smooth, Px_smooth = kfd
    @unpack μ, Λ, R = wks

    NT = size(EY, 1)
    i_NT = one(T) / NT

    μb = view(μ, yinds)
    Λb_e = view(Λ, yinds, xinds_estim)
    Λb_g = view(Λ, yinds, xinds_given)
    Yb = view(EY, :, yinds)
    Xb_e = transpose(view(x_smooth, xinds_estim, :))
    PXb_ee = view(Px_smooth, xinds_estim, xinds_estim, :)
    Xb_g = transpose(view(x_smooth, xinds_given, :))
    PXb_ge = view(Px_smooth, xinds_given, xinds_estim, :)

    # prep the system matrix
    mul!(XTX, transpose(Xb_e), Xb_e)
    sum!(XTX, PXb_ee, init=false)

    # prep the right-hand-side
    mul!(YTX, transpose(Yb), Xb_e)

    # compute the contributions from factors with given loadings
    mul!(XTX_ge, transpose(Xb_g), Xb_e)
    sum!(XTX_ge, PXb_ge, init=false)

    # subtract known contributions from Y
    mul!(YTX, Λb_g, XTX_ge, -1.0, 1.0)

    # are we estimating the mean? 
    if mean_estim
        # if we are, use its Schur complement to eliminate it from the system
        sum!(SX, transpose(Xb_e))
        BLAS.ger!(-i_NT, SX, SX, XTX)
        sum!(transpose(SY), Yb)
        BLAS.ger!(-i_NT, SY, SX, YTX)
    else
        # otherwise, subtract from Y
        copyto!(SY, μb)
        sum!(SX, transpose(Xb_e))
        BLAS.ger!(-i_NT, SY, SX, YTX)
    end

    # solve the system (using Cholesky, since matrix is SPD)
    copyto!(new_Λ, YTX)     # keep YTX safe, in case we need it below
    cXTX = cholesky!(Symmetric(XTX))  # overwrites XTX
    rdiv!(new_Λ, cXTX)

    _apply_constraint!(new_Λ, constraint, cXTX, view(R, yinds, yinds))

    if mean_estim
        # backward substitution
        copyto!(μb, SY)
        mul!(μb, new_Λ, SX, -i_NT, i_NT)
    end

    copyto!(Λb_e, new_Λ)

    return Λb_e
end




##################################################################################




struct EM_Observed_Covar_Wks{T}
    V::Vector{T}
    LP::Matrix{T}
end

function em_observed_covar_wks(M::DFM, wks::DFMKalmanWks{T}) where {T}
    NO = Kalman.kf_length_y(M)
    NS = Kalman.kf_length_x(M)
    LP = Matrix{T}(undef, NO, NS)
    V = Vector{T}(undef, NO)
    EM_Observed_Covar_Wks{T}(V, LP)
end

function _em_update_observed_covar!(R::AbstractMatrix{T}, EY, EX, PX, μ, Λ, V, LP) where {T}
    # new_R = 1/NT * (EY - 1*μᵀ - EX*Λᵀ)ᵀ(EY - 1*μᵀ - EX*Λᵀ)
    #    = 1/NT * 

    # new_R = 1/nobs * ( YᵀY - YᵀXΛᵀ - ΛXᵀY + ΛXᵀXΛᵀ + Λ sum(Px_smooth) Λᵀ )

    # In the case of have_mu = true, the YᵀX and XᵀX matrices have already been 
    #   corrected, but we still need to correct YᵀY.
    # Also, note that our XᵀX matrix already includes sum(Px_smooth)

    NT, NO = size(EY)
    NT, NS = size(EX)

    fill!(R, zero(T))
    for n = 1:NT
        V[:] = @view EY[n, :]
        V[:] -= μ
        mul!(V, Λ, @view(EX[n, :]), -1.0, 1.0)
        # R += V * transpose(V)
        BLAS.ger!(1.0, V, V, R)
    end
    for n = 1:NT
        mul!(LP, Λ, @view(PX[:, :, n]))
        mul!(R, LP, transpose(Λ), 1.0, 1.0)
    end
    ldiv!(NT, R)

    return R
end

function _em_update_observed_covar!(R::Diagonal{T}, EY, EX, PX, μ, Λ, V, LP) where {T}
    # new_R = 1/NT * (EY - 1*μᵀ - EX*Λᵀ)ᵀ(EY - 1*μᵀ - EX*Λᵀ)
    #    = 1/NT * 

    # new_R = 1/nobs * ( YᵀY - YᵀXΛᵀ - ΛXᵀY + ΛXᵀXΛᵀ + Λ sum(Px_smooth) Λᵀ )

    # In the case of have_mu = true, the YᵀX and XᵀX matrices have already been 
    #   corrected, but we still need to correct YᵀY.
    # Also, note that our XᵀX matrix already includes sum(Px_smooth)

    NT, NO = size(EY)
    NT, NS = size(EX)

    dR = R.diag

    fill!(dR, zero(T))
    for n = 1:NT
        V[:] = @view EY[n, :]
        V[:] -= μ
        mul!(V, Λ, @view(EX[n, :]), -1.0, 1.0)
        for i = 1:NO
            dR[i] += V[i] * V[i]
        end
    end
    for n = 1:NT
        mul!(LP, Λ, @view(PX[:, :, n]))
        for i = 1:NO
            dR[i] += @views BLAS.dot(LP[i, :], Λ[i, :])
        end
    end
    ldiv!(NT, R)
    return R
end

function em_update_observed_covar!(wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, em_wks::EM_Observed_Covar_Wks) where {T}
    @unpack μ, Λ, R = wks
    @unpack V, LP = em_wks
    @unpack x_smooth, Px_smooth = kfd
    EX = transpose(x_smooth)
    PX = Px_smooth
    return _em_update_observed_covar!(R, EY, EX, PX, μ, Λ, V, LP)
end



##################################################################################



struct EM_Transition_Block_Wks{T,TA,TQ}
    xinds_1::Vector{Int}
    xinds_2::Vector{Int}
    covar_estim::Bool
    constraint::Union{Nothing,DFMConstraint{T}}
    new_A::TA
    new_Q::TQ
    XTX_22::Matrix{T}
    XTX_12::Matrix{T}
end

function em_transition_block_wks(cn::Symbol, M::DFM, wks::DFMKalmanWks{T}) where {T}
    @unpack model, params = M
    @unpack A, Q = wks
    cb = model.components[cn]
    NS = nstates(cb)
    ord = DFMModels.order(cb)
    xinds = append!(Int[], indexin(states_with_lags(cb), states_with_lags(model)))
    xinds_1 = xinds[end-NS+1:end]
    xinds_2 = xinds[end-ord*NS+1:end]
    vA = view(A, xinds_1, xinds_2)
    ncons = length(vA) - sum(isnan, vA)
    if ncons > 0
        W = spzeros(ncons, length(vA))
        q = spzeros(ncons)
        con = 0
        for (i, v) = enumerate(vA)
            if !isnan(v)
                con = con + 1
                W[con, i] = 1
                q[con] = v
            end
        end
        @assert con == ncons
    else
        W = spzeros(0, 0)
        q = spzeros(0)
    end
    NS_1 = length(xinds_1)
    NS_2 = length(xinds_2)
    new_A = Matrix{T}(undef, NS_1, NS_2)
    new_Q = similar(get_covariance(cb, params[cn]))
    XTX_22 = Matrix{T}(undef, NS_2, NS_2)
    XTX_12 = Matrix{T}(undef, NS_1, NS_2)
    covar_estim = any(isnan, view(Q, xinds_1, xinds_1))
    constraint = DFMSolver.DFMConstraint(length(xinds_2), W, q)
    return EM_Transition_Block_Wks{T,typeof(new_A),typeof(new_Q)}(
        xinds_1, xinds_2, covar_estim,
        constraint, new_A, new_Q, XTX_22, XTX_12
    )
end

function em_update_transition_block!(wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix{T}, em_wks::EM_Transition_Block_Wks{T}) where {T}
    # unpack the inputs
    @unpack xinds_1, xinds_2, covar_estim, constraint = em_wks
    @unpack new_A, new_Q, XTX_22, XTX_12 = em_wks
    @unpack A, Q = wks
    @unpack x_smooth, Px_smooth, Pxx_smooth = kfd

    vQ = view(Q, xinds_1, xinds_1)

    EXm_2 = transpose(@view x_smooth[xinds_2, begin:end-1])
    EXp_1 = transpose(@view x_smooth[xinds_1, begin+1:end])

    ####
    # new_A = XpᵀXm / XmᵀXm

    # construct XmᵀXm
    mul!(XTX_22, transpose(EXm_2), EXm_2)
    sum!(XTX_22, @view(Px_smooth[xinds_2, xinds_2, begin:end-1]), init=false)

    # construct XpᵀXm
    mul!(XTX_12, transpose(EXp_1), EXm_2)
    sum!(transpose(XTX_12), @view(Pxx_smooth[xinds_2, xinds_1, begin:end-1]), init=false)

    # solve for new_A
    cXTX = cholesky!(Symmetric(XTX_22))
    copyto!(new_A, XTX_12)
    rdiv!(new_A, cXTX)

    _apply_constraint!(new_A, constraint, cXTX, vQ)

    A[xinds_1, xinds_2] = new_A

    if covar_estim
        PXp_1 = @view(Px_smooth[xinds_1, xinds_1, begin+1:end])
        _em_update_transition_covar!(new_Q, EXp_1, PXp_1, new_A, XTX_12)
        copyto!(vQ, new_Q)
    end

    return new_A, new_Q
end

function _em_update_transition_covar!(Q::Diagonal{T}, EX::AbstractMatrix, PX::AbstractArray, A::AbstractMatrix, PXpm::AbstractMatrix) where {T}
    NT, NS = size(EX)
    dQ = Q.diag
    fill!(dQ, zero(T))
    for n = 1:NT
        for i = 1:NS
            dQ[i] += EX[n, i] * EX[n, i] + PX[i, i, n]
        end
    end
    for i = 1:NS
        dQ[i] -= BLAS.dot(A[i, :], PXpm[i, :])
    end
    ldiv!(NT, dQ)
    return Q
end

function _em_update_transition_covar!(Q::Symmetric{T}, EX::AbstractMatrix, PX::AbstractArray, A::AbstractMatrix, PXpm::AbstractMatrix) where {T}
    NT = size(EX, 1)
    dQ = Q.data
    mul!(dQ, transpose(EX), EX)
    sum!(dQ, PX, init=false)
    mul!(dQ, A, transpose(PXpm), -1, 1)
    ldiv!(NT, dQ)
    return Q
end



##################################################################################



struct EM_Observed_Wks{T}
    loadings::Vector{EM_Observed_Block_Loading_Wks{T}}
    covars::EM_Observed_Covar_Wks
end
function em_observed_wks(M::DFM, wks::DFMKalmanWks{T}) where {T}
    @unpack model, params = M
    loadings = EM_Observed_Block_Loading_Wks{T}[]
    for on in keys(model.observed)
        push!(loadings, em_observed_block_loading_wks(on, M, wks))
    end
    covars = em_observed_covar_wks(M, wks)
    return EM_Observed_Wks{T}(loadings, covars)
end


struct EM_Transition_Wks{T}
    blocks::Vector{EM_Transition_Block_Wks{T}}
end
function em_transition_wks(M::DFM, wks::DFMKalmanWks{T}) where {T}
    @unpack model, params = M
    blocks = EM_Transition_Block_Wks{T}[]
    for cn in keys(model.components)
        push!(blocks, em_transition_block_wks(cn, M, wks))
    end
    return EM_Transition_Wks{T}(blocks)
end


struct EM_Wks{T}
    observed::EM_Observed_Wks{T}
    transition::EM_Transition_Wks{T}
end

function em_workspace(M::DFM, wks::DFMKalmanWks{T}) where {T}
    obs = em_observed_wks(M, wks)
    trans = em_transition_wks(M, wks)
    return EM_Wks{T}(obs, trans)
end



##################################################################################



function EMstep!(wks::DFMKalmanWks, x0::AbstractVector, Px0::AbstractMatrix,
    kfd::Kalman.AbstractKFData, EY::AbstractMatrix, M::DFM, em_wks::EM_Wks)
    @unpack observed, transition = em_wks
    for em_w in observed.loadings
        em_update_observed_block_loading!(wks, kfd, EY, em_w)
    end
    em_update_observed_covar!(wks, kfd, EY, observed.covars)
    for em_w in transition.blocks
        em_update_transition_block!(wks, kfd, EY, em_w)
    end
    x0 .= kfd.x_smooth[:,begin]
    for (cb, tr_wks) in zip(values(M.model.components),em_wks.transition.blocks)
        xinds_1 = tr_wks.xinds_1
        if cb isa IdiosyncraticComponents
            Px0[xinds_1, xinds_1] = Diagonal(kfd.Px_smooth[xinds_1, xinds_1, 1])
        else
            Px0[xinds_1, xinds_1] = kfd.Px_smooth[xinds_1, xinds_1, 1]
        end
    end
    return wks
end



##################################################################################



function EMestimate(M::DFM, Y::AbstractMatrix,
    wks::DFMKalmanWks{T}=DFMKalmanWks(M),
    x0::AbstractVector=zeros(Kalman.kf_length_x(M)),
    Px0::AbstractMatrix=Matrix{T}(1e-10 * I(Kalman.kf_length_x(M)))
    ;
    initial_guess::Union{Nothing,AbstractVector}=nothing,
    maxiter=100, rftol=1e-4, axtol=1e-4,
    verbose=false,
    anymissing::Bool=any(isnan, Y)
) where {T}
    @unpack model, params = M

    if !any(isnan, params)
        @error "No parameters have been marked NaN for estimation."
        return
    end

    em_wks = em_workspace(M, wks)

    kfd = Kalman.KFDataSmoother(size(Y, 1), M, Y, wks)
    kf = Kalman.KFilter(kfd)

    old_params = copy(params)

    EY = copy(Y)

    if any(isnan, EY)
        error("I can't do this!")
        # _impute_using_splines(EY, Y)
    end

    if isnothing(initial_guess)
        EMinit!(wks, kfd, Y, M, em_wks)
        _update_params!(params, model, wks)
    else
        copyto!(params, initial_guess)
    end
    _update_wks!(wks, model, params)


    loglik = -Inf
    for iter = 1:maxiter

        Kalman.dk_filter!(kf, Y, wks, x0, Px0, false, anymissing)
        Kalman.dk_smoother!(kf, Y, wks)
        em_impute_observed!(EY, kfd, Y)
        EMstep!(wks, x0, Px0, kfd, EY, M, em_wks)

        copyto!(old_params, params)
        _update_params!(params, model, wks)
        _update_wks!(wks, model, params)

        loglik_new = sum(kfd.loglik)
        dx = maximum(abs, params - old_params)
        df = 2 * abs(loglik - loglik_new) / (abs(loglik) + abs(loglik_new) + eps())

        if verbose == true && mod(iter, 10) == 0
            sl = @sprintf "%.6g" loglik
            sx = @sprintf "%.6g" dx
            sf = @sprintf "%.6g" df
            @info "EM iteration $(lpad(iter, 5)): loglik=$sl, df=$sf, dx=$sx"
        end

        loglik = loglik_new

        if df < rftol || dx < axtol
            break
        end
    end
    return
end

function EMinit!(wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, M::DFM, em_wks::EM_Wks) where {T}
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
    SY = svd(resY)
    FALL = view(SY.U, :, 1:NF)
    rmul!(FALL, Diagonal(view(SY.S, 1:NF)))

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
        if any(isnan, view(Λ, :, xinds_1))
            # compute initial loadings
            for (obs_wks, (on, ob)) in zip(em_wks.observed.loadings, model.observed)
                haskey(ob.components, cn) || continue # skip components that are not loaded by this observed block
                @unpack yinds, constraint = obs_wks
                any(isnan, view(Λ, yinds, xinds_1)) || continue # skip if loading is given
                NC = DFMModels.mf_ncoefs(ob)
                xinds_2 = last(xinds_1) .+ (1-nf*NC:0)
                L = Matrix{T}(undef, length(yinds), nf * NC)
                FTF = Matrix{T}(undef, nf * NC, nf * NC)
                bY = view(resY, NL:NT, yinds)
                bFT = view(FT, xinds_2, :)
                mul!(transpose(L), bFT, bY)
                mul!(FTF, bFT, transpose(bFT))
                cFTF = cholesky!(Symmetric(FTF))
                rdiv!(L, cFTF)
                _apply_constraint!(L, constraint, cFTF, cov(bY))
                for (j, x) in enumerate(xinds_2)
                    for (i, y) in enumerate(yinds)
                        if isnan(Λ[y, x])
                            Λ[y, x] = L[i, j]
                        end
                    end
                end
            end
        end
        # subtract contributions of factors
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
        for (j, x2) in enumerate(xinds_2)
            for (i, x1) in enumerate(xinds_1)
                if isnan(A[x1, x2])
                    A[x1, x2] = FRHS[i, j]
                end
            end
        end
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
            for (i, xi) in enumerate(xinds_1)
                for (j, xj) in enumerate(xinds_1)
                    if isnan(Q[xi, xj])
                        Q[xi, xj] = COV[i, j]
                    end
                end
            end
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

    nanR = map(isnan, wks.R)
    wks.R[nanR] .= I(NO)[nanR]

    return wks
end


