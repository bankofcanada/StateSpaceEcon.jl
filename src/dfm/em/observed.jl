##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

struct EM_Observed_Block_Loading_Wks{T,YI<:AbstractVector{Int},XIE<:AbstractVector{Int},XIG<:AbstractVector{Int}}
    yinds::YI
    xinds_estim::XIE
    xinds_given::XIG
    mean_estim::Bool
    orthogonal_factors_ics::Bool
    constraint::Union{Nothing,DFMConstraint{T}}
    # pre-allocated matrices
    YTX::Matrix{T}
    new_Λ::Matrix{T}
    XTX::Matrix{T}
    XTX_ge::Matrix{T}
    SY::Vector{T}
    SX::Vector{T}
    # indices corresponding to each block
    inds_cb::Vector{UnitRange{Int}}

    function EM_Observed_Block_Loading_Wks{T}(yinds, xinds_estim, xinds_given, args...) where {T}
        ry = UnitRange(extrema(yinds)...)
        yinds == ry && (yinds = ry)
        rxe = UnitRange(extrema(xinds_estim)...)
        xinds_estim == rxe && (xinds_estim = rxe)
        rxg = UnitRange(extrema(xinds_given, init=(typemax(Int), typemin(Int)))...)
        xinds_given == rxg && (xinds_given = rxg)
        return new{T,typeof(yinds),typeof(xinds_estim),typeof(xinds_given)}(yinds, xinds_estim, xinds_given, args...)
    end

end


function em_observed_block_loading_wks(on::Symbol, M::DFM, wks::DFMKalmanWks{T};
    orthogonal_factors_ics=true
) where {T}
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
    inds_cb = UnitRange{Int}[]
    global_offset = 0
    local_offset = 0
    W = spzeros(0, 0)
    q = spzeros(0)
    for (cn, cb) in ob.components
        bcols = global_offset .+ (1:nstates_with_lags(cb))
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
        elseif orthogonal_factors_ics && (cb isa IdiosyncraticComponents)
            # loadings of idiosyncratic components are identity matrix, 
            nothing
        else
            # loadings of this block are given. 
            # Record the column indices of columns that are not all zero
            for c in bcols
                xc = xinds[c]
                if !iszero(view(Λ, yinds, xc))
                    push!(xinds_given, xc)
                end
            end
        end
        push!(inds_cb, local_offset+1:length(xinds_estim))
        local_offset = length(xinds_estim)
        global_offset = last(bcols)
    end
    nest = length(xinds_estim)
    ngiv = length(xinds_given)
    nobs = length(yinds)
    return EM_Observed_Block_Loading_Wks{T}(
        yinds,
        # xinds, 
        xinds_estim, xinds_given,
        mean_estim, orthogonal_factors_ics,
        DFMConstraint(nest, W, q),
        Matrix{T}(undef, nobs, nest),     # YTX
        Matrix{T}(undef, nobs, nest),     # new_Λ
        Matrix{T}(undef, nest, nest),     # XTX
        Matrix{T}(undef, ngiv, nest),     # XTX_ge
        Vector{T}(undef, nobs),           # SY
        Vector{T}(undef, nest),           # SX
        inds_cb,
    )
end

@generated function em_update_observed_block_loading!(wks::DFMKalmanWks, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, em_wks::EM_Observed_Block_Loading_Wks, ::Val{anymissing}, ::Val{use_full_XTX}) where {anymissing,use_full_XTX}
    if anymissing
        return quote
            have_nans = any(isnan, view(EY, :, em_wks.yinds))
            _em_update_observed_block_loading!(wks, kfd, EY, em_wks, Val(have_nans), Val(use_full_XTX),)
        end
    else
        return quote
            _em_update_observed_block_loading!(wks, kfd, EY, em_wks, Val(false), Val(use_full_XTX),)
        end
    end
end

function _add_kron!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, alpha=1, beta=1)
    size(C) == LinearAlgebra._kronsize(A, B) || throw(DimensionMismatch("kron!"))
    m = firstindex(C)
    for j in axes(A, 2), l in axes(B, 2), i in axes(A, 1)
        @inbounds Aij = beta * A[i, j]
        for k in axes(B, 1)
            @inbounds C[m] = alpha * C[m] + Aij * B[k, l]
            m += 1
        end
    end
    return C
end

# method for missing values (one observed at a time, no Kronecker!)
function _em_update_observed_block_loading!(
    wks::DFMKalmanWks{T},
    kfd::Kalman.AbstractKFData,
    EY::AbstractMatrix,
    em_wks::EM_Observed_Block_Loading_Wks,
    ::Val{true},
    ::Val{use_full_XTX}, # controls whether to use the full XTX matrix when enforcing constraint
) where {T,use_full_XTX}
    # unpack the inputs
    @unpack yinds, xinds_estim, xinds_given, constraint = em_wks
    @unpack mean_estim, orthogonal_factors_ics = em_wks
    @unpack new_Λ, YTX, XTX, XTX_ge, SX, SY = em_wks
    @unpack x_smooth, Px_smooth = kfd
    @unpack μ, Λ, R = wks

    NT = size(EY, 1)
    NY = length(yinds)

    μb = view(μ, yinds)
    Λb_e = view(Λ, yinds, xinds_estim)

    constraint_and_missing_XTX = !isnothing(constraint) && !use_full_XTX
    NLC = length(new_Λ)
    A = constraint_and_missing_XTX ? zeros(NLC, NLC) : nothing
    invR = constraint_and_missing_XTX ? inv(R) : nothing

    mask = falses(NT)
    for i in 1:NY
        yi = yinds[i]
        μi = μ[yi]

        for j = eachindex(mask)
            @inbounds mask[j] = !isnan(EY[j, yi])
        end

        i_NT = one(T) / sum(mask)

        Yi = view(EY, mask, yi)
        Xi = transpose(view(x_smooth, xinds_estim, mask))
        PXi = view(Px_smooth, xinds_estim, xinds_estim, mask)

        YiTX = view(YTX, i:i, :)

        # prep the system matrix
        mul!(XTX, transpose(Xi), Xi)
        sum!(XTX, PXi, init=false)
        # prep the right-hand-side
        mul!(YiTX, transpose(Yi), Xi)

        # are there any components whose loadings are known?
        if length(xinds_given) > 0
            # if there are, subtract their contributions from the right hand side
            Xi_g = transpose(view(x_smooth, xinds_given, mask))
            PXi_g = view(Px_smooth, xinds_given, xinds_estim, mask)
            Λb_g = view(Λ, yi:yi, xinds_given)

            # compute the contributions from factors with given loadings
            mul!(XTX_ge, transpose(Xi_g), Xi)
            sum!(XTX_ge, PXi_g, init=false)

            # subtract known contributions from Y
            mul!(YiTX, Λb_g, XTX_ge, -1.0, 1.0)
        end

        # are we estimating the mean? 
        if mean_estim
            # if we are, use its Schur complement to eliminate it from the system
            sum!(SX, transpose(Xi))
            BLAS.ger!(-i_NT, SX, SX, XTX)
            SY[i] = sum(Yi)
            YiTX[:] -= i_NT * SY[i] * SX
        else
            # otherwise, subtract known mean from Y
            SY[i] = μi
            if !iszero(μi)
                sum!(SX, transpose(Xi))
                YiTX[:] -= μi * SX
            end
        end
        
        if constraint_and_missing_XTX
            iR = invR[i,i]
            A[i:NY:end, i:NY:end] .= XTX .* iR
        end

        # solve the system (using Cholesky factorization, since matrix is SPD)
        new_Λ[i, :] = YiTX
        cXTX = cholesky!(Symmetric(XTX))  # overwrites XTX
        rdiv!(view(new_Λ, i:i, :), cXTX)
    end

    # apply constraints, if any
    if constraint_and_missing_XTX
        @unpack W, q, BT, Tcc, Tc = constraint
        Tc[:] = q
        mul!(Tc, W, vec(new_Λ), -1.0, 1.0)
        # A is actually the inverse of what we need
        Kalman._symm!(A)
        @static if true
            cA = cholesky!(Symmetric(A))
            BT[:, :] = W
            rdiv!(BT, cA)
        else
            LAPACK.potrf!('U', A)
            LAPACK.potri!('U', A)
            mul!(BT, W, Symmetric(A, :U))
        end
        mul!(Tcc, BT, transpose(W))
        @static if true
            cTcc = qr!(Tcc, ColumnNorm())
            ldiv!(cTcc, Tc)
        else
            LAPACK.potrf!('U', Tcc)
            LAPACK.potrs!('U', Tcc, Tc)
        end
        mul!(vec(new_Λ), transpose(BT), Tc, 1, 1)
    elseif !isnothing(constraint)
        Xi = transpose(view(x_smooth, xinds_estim, mask))
        PXi = view(Px_smooth, xinds_estim, xinds_estim, mask)
        mul!(XTX, transpose(Xi), Xi)
        sum!(XTX, PXi, init=false)
        if mean_estim
            # if we are, use its Schur complement to eliminate it from the system
            sum!(SX, transpose(Xi))
            BLAS.ger!(-one(T) / NT, SX, SX, XTX)
        end
        cXTX = cholesky!(Symmetric(XTX))
        _apply_constraint!(new_Λ, constraint, cXTX, view(R, yinds, yinds))
    end

    if mean_estim
        # solve for mean using backward substitution
        for i = 1:NY
            yi = yinds[i]
            map!(!isnan, mask, view(EY, :, yi))
            Xi = transpose(view(x_smooth, xinds_estim, mask))
            i_NT = one(T) / sum(mask)
            sum!(SX, transpose(Xi))
            μb[i] = (SY[i] - dot(view(new_Λ, i, :), SX)) * i_NT
        end
    end

    copyto!(Λb_e, new_Λ)

    return Λb_e

end

# methods when there are no missing values (faster linalg)
function _em_update_observed_block_loading!(
    wks::DFMKalmanWks{T},
    kfd::Kalman.AbstractKFData,
    EY::AbstractMatrix,
    em_wks::EM_Observed_Block_Loading_Wks,
    ::Val{false},
    ::Val, # {use_full_XTX} not relevant here, only when there are missing values
) where {T}
    # unpack the inputs
    @unpack yinds, xinds_estim, xinds_given, constraint = em_wks
    @unpack mean_estim, orthogonal_factors_ics = em_wks
    @unpack new_Λ, YTX, XTX, XTX_ge, SX, SY = em_wks
    @unpack x_smooth, Px_smooth = kfd
    @unpack μ, Λ, R = wks

    NT = size(EY, 1)
    i_NT = one(T) / NT

    μb = view(μ, yinds)
    Λb_e = view(Λ, yinds, xinds_estim)
    Yb = view(EY, :, yinds)
    Xb_e = transpose(view(x_smooth, xinds_estim, :))
    PXb_ee = view(Px_smooth, xinds_estim, xinds_estim, :)

    # prep the system matrix
    mul!(XTX, transpose(Xb_e), Xb_e)
    sum!(XTX, PXb_ee, init=false)

    # prep the right-hand-side
    mul!(YTX, transpose(Yb), Xb_e)

    if length(xinds_given) > 0
        Λb_g = view(Λ, yinds, xinds_given)
        Xb_g = transpose(view(x_smooth, xinds_given, :))
        PXb_ge = view(Px_smooth, xinds_given, xinds_estim, :)

        # compute the contributions from factors with given loadings
        mul!(XTX_ge, transpose(Xb_g), Xb_e)
        sum!(XTX_ge, PXb_ge, init=false)

        # subtract known contributions from Y
        mul!(YTX, Λb_g, XTX_ge, -1.0, 1.0)
    end

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
        BLAS.ger!(-1.0, SY, SX, YTX)
    end

    # solve the system (using Cholesky factorization, since matrix is SPD)
    copyto!(new_Λ, YTX)     # keep YTX safe, in case we need it below
    cXTX = cholesky!(Symmetric(XTX))  # overwrites XTX
    rdiv!(new_Λ, cXTX)

    _apply_constraint!(new_Λ, constraint, cXTX, view(R, yinds, yinds))

    if mean_estim
        # solve for mean using backward substitution
        copyto!(μb, SY)
        mul!(μb, new_Λ, SX, -i_NT, i_NT)
    end

    copyto!(Λb_e, new_Λ)

    return Λb_e
end




##################################################################################




struct EM_Observed_Covar_Wks{T}
    covar_estim::Bool
    V::Vector{T}
    LP::Matrix{T}
end

function em_observed_covar_wks(M::DFM, wks::DFMKalmanWks{T}) where {T}
    covar_estim = any(isnan, wks.R)
    NO = Kalman.kf_length_y(M)
    NS = Kalman.kf_length_x(M)
    LP = Matrix{T}(undef, NO, NS)
    V = Vector{T}(undef, NO)
    EM_Observed_Covar_Wks{T}(covar_estim, V, LP)
end

function _em_update_observed_covar!(R::AbstractMatrix{T}, EY, EX, PX, μ, Λ, V, LP, anymissing::Val{true}) where {T}
    NT, NO = size(EY)
    NS = size(EX, 2)

    old_R = copy(R)
    fill!(R, zero(T))
    R1 = zeros(size(R))
    for n = 1:NT
        Yn = view(EY, n, :)
        all(isnan, Yn) && continue
        mul!(LP, Λ, @view(PX[:, :, n]))
        mul!(R1, LP, transpose(Λ))
        V .= Yn
        V .-= μ
        mul!(V, Λ, view(EX, n, :), -1.0, 1.0)
        BLAS.ger!(1.0, V, V, R1)
        if R isa Diagonal
            for i = 1:NO
                R[i, i] += isnan(R1[i, i]) ? old_R[i, i] : R1[i, i]
            end
        else
            for i = 1:NO
                for j = i:NO
                    R[i, j] += isnan(R1[i, j]) ? old_R[i, j] : R1[i, j]
                end
            end
        end
    end
    ldiv!(NT, R)
    if !(R isa Diagonal)
        for i = 1:NO
            for j = 1:i-1
                R[i, j] = R[j, i]
            end
        end
    end
    return R
end

function _em_update_observed_covar!(R::AbstractMatrix{T}, EY, EX, PX, μ, Λ, V, LP, anymissing::Val{false}) where {T}
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

function _em_update_observed_covar!(R::Diagonal{T}, EY, EX, PX, μ, Λ, V, LP, anymissing::Val{false}) where {T}
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

function em_update_observed_covar!(wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, em_wks::EM_Observed_Covar_Wks, anymissing::Val) where {T}
    @unpack covar_estim, V, LP = em_wks
    @unpack μ, Λ, R = wks
    covar_estim || return R
    @unpack x_smooth, Px_smooth = kfd
    EX = transpose(x_smooth)
    PX = Px_smooth
    return _em_update_observed_covar!(R, EY, EX, PX, μ, Λ, V, LP, anymissing)
end

##################################################################################


struct EM_Observed_Wks{T,LT}
    loadings::LT
    covars::EM_Observed_Covar_Wks
end
function em_observed_wks(M::DFM, wks::DFMKalmanWks{T}; orthogonal_factors_ics=true) where {T}
    @unpack model, params = M
    loadings = (; (nm => em_observed_block_loading_wks(nm, M, wks; orthogonal_factors_ics) for nm in keys(model.observed))...)
    covars = em_observed_covar_wks(M, wks)
    return EM_Observed_Wks{T,typeof(loadings)}(loadings, covars)
end

