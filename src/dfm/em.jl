##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

"""
    em_impute_kalman!(EY, Y, kfd)

Fill in any missing values with their expected values according to the 
Kalman smoother.

* `Y` is the original data with `NaN` values where data is missing.
* `kfd` is the KalmanFilterData instance containing the output from Kalman
  filter and Kalman smoother.
* `EY` is modified in place. Where `Y` contains a `NaN`, the corresponding 
  entries in `EY` are imputed. The rest of `EY` is not modified.

NOTE: It is assumed that `EY` equals `Y` everywhere where `Y` is not `NaN`, 
however, this is neither checked nor enforced.
"""
function em_impute_kalman!(EY::AbstractMatrix{T}, Y::AbstractMatrix{T}, kfd::Kalman.AbstractKFData) where {T<:AbstractFloat}
    EY === Y && return EY
    # @assert EY[.!isnan.(Y)] == Y[.!isnan.(Y)]
    YS = kfd.y_smooth  # this one is transposed (NO × NT)
    for i = axes(Y, 1)
        for j = axes(Y, 2)
            @inbounds yij = Y[i, j]
            isnan(yij) || continue  # EY is a copy of Y. The non-NaN values never change
            EY[i, j] = YS[j, i] # update NaN value
        end
    end
    return EY
end

"""
    em_impute_interp!(EY, Y, IT)

Fill in any missing values using interpolation.

* `Y` is the original data with `NaN` values where data is missing.
* `IT` is an instance of `Interpolations.InterpolationType`. Default is
  `AkimaMonotonicInterpolation`. See documentation of `Interpolations.jl`
  package for details and other interpolation choices.
* `EY` is modified in place. Where `Y` contains a `NaN`, the corresponding 
  entries in `EY` are imputed. The rest of `EY` is not modified.

NOTE: It is assumed that `EY` equals `Y` everywhere where `Y` is not `NaN`, 
however, this is neither checked nor enforced.
"""
function em_impute_interpolation!(EY::AbstractMatrix{T}, Y::AbstractMatrix{T},
    IT::Interpolations.InterpolationType=Interpolations.FritschCarlsonMonotonicInterpolation(),
    k::Int=3
) where {T<:AbstractFloat}
    EY === Y && return EY
    rows, cols = axes(EY)
    valid_number = similar(Y, Bool, rows) # `true` where Y is not NaN
    tmp = zeros(T, rows)
    for j in cols
        tmp .= Y[:, j]
        EYj = view(EY, :, j)
        for i in rows
            @inbounds valid_number[i] = !isnan(tmp[i])
        end
        all(valid_number) && continue
        # use cubic interpolation between the first and last non-NaN
        i1 = findfirst(valid_number)
        i2 = findlast(valid_number)
        interp = interpolate(view(rows, valid_number), view(Y, valid_number, j), IT)
        for i in i1:i2
            if @inbounds !valid_number[i]
                val = interp(i)
                @inbounds EYj[i] = tmp[i] = val
            end
        end
        # use centered 2k+1 moving average for leading and trailing NaNs
        # where NaNs remain, use median to compute the moving average
        ym = nanmedian(tmp)
        for i = first(rows):i1-1
            tmp[i] = ym
        end
        for i = i2+1:last(rows)
            tmp[i] = ym
        end
        for i = Iterators.flatten((first(rows):i1-1, i2+1:last(rows)))
            i1 = max(first(rows), i - k)
            a1 = max(0, k - i + 1)
            i2 = min(last(rows), i + k)
            a2 = max(0, i + k - last(rows))
            val = (sum(i -> tmp[i], i1:i2) + a1 * tmp[begin] + a2 * tmp[end]) / (2k + 1)
            @inbounds EYj[i] = val
        end
    end
    return EY
end

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

@generated function em_update_observed_block_loading!(wks::DFMKalmanWks, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, em_wks::EM_Observed_Block_Loading_Wks, ::Val{anymissing}) where {anymissing}
    if anymissing
        return quote
            have_nans = any(isnan, view(EY, :, em_wks.yinds))
            _em_update_observed_block_loading!(Val(have_nans), wks, kfd, EY, em_wks)
        end
    else
        return quote
            _em_update_observed_block_loading!(Val(false), wks, kfd, EY, em_wks)
        end
    end
end

# methods for missing values (one observed at a time, no Kronecker!)
function _em_update_observed_block_loading!(::Val{true}, wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, em_wks::EM_Observed_Block_Loading_Wks) where {T}
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
    Λb_g = view(Λ, yinds, xinds_given)

    mask = falses(NT)
    for i in 1:NY
        yi = yinds[i]
        μi = μ[yi]
        map!(!isnan, mask, view(EY, :, yi))
        Yi = view(EY, mask, yi)
        Xi = transpose(view(x_smooth, xinds_estim, mask))
        PXi = view(Px_smooth, xinds_estim, xinds_estim, mask)

        i_NT = one(T) / sum(mask)

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

        # solve the system (using Cholesky factorization, since matrix is SPD)
        new_Λ[i, :] = YiTX
        cXTX = cholesky!(Symmetric(XTX))  # overwrites XTX
        rdiv!(view(new_Λ, i:i, :), cXTX)
    end

    # apply constraints, if any
    if !isnothing(constraint)
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
function _em_update_observed_block_loading!(::Val{false}, wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData, EY::AbstractMatrix, em_wks::EM_Observed_Block_Loading_Wks) where {T}
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
        for i = 1:NO
            for j = i:NO
                R[i, j] += isnan(R1[i, j]) ? old_R[i, j] : R1[i, j]
                R isa Diagonal && break
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



struct EM_Transition_Block_Wks{T,TA,TQ}
    xinds::Vector{Int}
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
        xinds, xinds_1, xinds_2, covar_estim,
        constraint, new_A, new_Q, XTX_22, XTX_12
    )
end

function em_update_transition_block!(wks::DFMKalmanWks{T}, kfd::Kalman.AbstractKFData,
    EY::AbstractMatrix{T}, em_wks::EM_Transition_Block_Wks{T},
    ::Val{use_x0_smooth}
) where {T,use_x0_smooth}
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

    if use_x0_smooth
        x0s = kfd.x0_smooth[xinds_2]
        BLAS.ger!(1.0, x0s, x0s, XTX_22)
        XTX_22 .+= kfd.Px0_smooth[xinds_2, xinds_2, 1]
        BLAS.ger!(1.0, kfd.x_smooth[xinds_1, 1], x0s, XTX_12)
        XTX_12 .+= transpose(kfd.Pxx0_smooth[xinds_2, xinds_1])
    end

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


struct EM_Transition_Wks{T,TBLK}
    blocks::TBLK
end
function em_transition_wks(M::DFM, wks::DFMKalmanWks{T}) where {T}
    @unpack model, params = M
    blocks = (; (nm => em_transition_block_wks(nm, M, wks) for nm in keys(model.components))...)
    return EM_Transition_Wks{T,typeof(blocks)}(blocks)
end


struct EM_Wks{T,TOBS,TTRS}
    observed::TOBS
    transition::TTRS
end

function em_workspace(M::DFM, wks::DFMKalmanWks{T}; orthogonal_factors_ics=true) where {T}
    obs = em_observed_wks(M, wks; orthogonal_factors_ics)
    trans = em_transition_wks(M, wks)
    return EM_Wks{T,typeof(obs),typeof(trans)}(obs, trans)
end



##################################################################################



function EMstep!(wks::DFMKalmanWks{T}, x0::AbstractVector{T}, Px0::AbstractMatrix{T},
    kfd::Kalman.AbstractKFData, EY::AbstractMatrix{T}, M::DFM, em_wks::EM_Wks{T},
    anymissing::Bool, # =any(isnan, EY)
    use_x0_smooth::Bool=false
) where {T}
    @unpack observed, transition = em_wks
    for em_w in observed.loadings
        em_update_observed_block_loading!(wks, kfd, EY, em_w, Val(anymissing))
    end
    em_update_observed_covar!(wks, kfd, EY, observed.covars, Val(anymissing))
    for em_w in transition.blocks
        em_update_transition_block!(wks, kfd, EY, em_w, Val(use_x0_smooth))
    end
    x0[:] = kfd.x0_smooth
    # fill!(x0, zero(T))
    for (cb, tr_wks) in zip(values(M.model.components), em_wks.transition.blocks)
        xinds = tr_wks.xinds
        if cb isa IdiosyncraticComponents
            # each idiosyncratic component may be autocorrelated with its lags, 
            # but they are independent of each other. 
            # So, copy the blocks for the individual components only
            NS = nstates(cb)
            xinds = xinds[1:NS:end]
            for _ = 1:NS
                Px0[xinds, xinds] = kfd.Px0_smooth[xinds, xinds]
                xinds .+= 1
            end
        else
            Px0[xinds, xinds] = kfd.Px0_smooth[xinds, xinds]
        end
    end
    return wks
end



##################################################################################


function _scale_model!(wks::DFMKalmanWks{T}, Mx, Wx, model::DFMModel, em_wks::EM_Wks) where {T}
    # IDEA: factors (CommonComponents) have loadings, so we scale the loadings
    #       obs noise and idiosyncratic components figure in with fixed loadings, 
    #       so we scale thier noises covariances instead.
    # scale the loadings of factors
    μ, Λ, R, Q = wks.μ, wks.Λ, wks.R, wks.Q
    for (obs_wks, oblk) in zip(em_wks.observed.loadings, values(model.observed))
        @unpack yinds, xinds_estim, inds_cb, constraint = obs_wks
        D = Diagonal(Wx[yinds])
        for (ind, (bnm, blk)) in zip(inds_cb, oblk.components)
            if blk isa IdiosyncraticComponents
                tmp = oblk.comp2vars[bnm]
                byinds = [!isa(tmp[x], DFMModels._NoCompRef) for x in observed(oblk)]
                xinds_1 = em_wks.transition.blocks[bnm].xinds_1
                ldiv!(Diagonal(Wx[yinds[byinds]]), view(Q, xinds_1, xinds_1))
            else
                ldiv!(D, view(Λ, yinds, xinds_estim[ind]))
            end
        end
        ldiv!(D, view(R, yinds, yinds))
        @unpack W, q = constraint
        nnz(q) == 0 && continue
        oblk_Λ = vec(view(Λ, yinds, xinds_estim))
        rows = rowvals(W)
        vals = nonzeros(W)
        fill!(q.nzval, zero(T))
        for col = 1:size(W, 2)
            for i in nzrange(W, col)
                row = rows[i]
                row in q.nzind || continue
                val = vals[i]
                q[row] += val * oblk_Λ[col]
            end
        end
    end
    fill!(μ, zero(T))
    return wks
end


function EMestimate!(M::DFM, Y::AbstractMatrix,
    wks::DFMKalmanWks{T}=DFMKalmanWks(M),
    x0::AbstractVector=zeros(Kalman.kf_length_x(M)),
    Px0::AbstractMatrix=Matrix{T}(1e-10 * I(Kalman.kf_length_x(M))),
    em_wks::EM_Wks{T}=em_workspace(M, wks),
    kfd=Kalman.KFDataSmoother(size(Y, 1), M, Y, wks),
    kf=Kalman.KFilter(kfd)
    ;
    fwdstate::Bool=false,
    initial_guess::Union{Nothing,AbstractVector}=nothing,
    maxiter=100, rftol=1e-4, axtol=1e-4,
    verbose=false,
    impute_missing::Bool=false,  # true - use Kalman smoother to impute missing data, false - treat missing data as in Banbura & Modugno 2014
    use_x0_smooth::Bool=false, # true - use x0_smooth in the EMstep update (experimental). Set to `false` for normal operation.
    anymissing::Bool=any(isnan, Y)
) where {T}
    @unpack model, params = M

    if !any(isnan, params)
        @error "No parameters have been marked NaN for estimation."
        return
    end

    org_params = copy(params)
    old_params = copy(params)

    # scale data
    Mx = nanmean(Y, dims=1)
    for i = eachindex(Mx)
        v = wks.μ[i]
        isnan(v) || (Mx[i] = v)
    end
    Wx = nanstd(Y, dims=1, mean=Mx)
    Y .= (Y .- Mx) ./ Wx


    if anymissing
        # make a copy of Y and impute using cubic splines
        # these imputed values are only used in EMinit! immediately below.
        EY = em_impute_interpolation!(copy(Y), Y, FritschCarlsonMonotonicInterpolation())
    else
        EY = Y # no copy, just reference
    end

    if isnothing(initial_guess)
        _scale_model!(wks, Mx, Wx, model, em_wks)
        EMinit!(wks, kfd, EY, M, em_wks)
    else
        _update_wks!(wks, model, initial_guess)
        _scale_model!(wks, Mx, Wx, model, em_wks)
        if any(isnan, initial_guess)
            EMinit!(wks, kfd, EY, M, em_wks)
        end
    end
    _update_params!(params, model, wks)

    if anymissing && !impute_missing
        copyto!(EY, Y)
    end

    loglik = -Inf
    loglik_best = -Inf
    iter_best = 0
    params_best = copy(params)
    for iter = 1:maxiter

        ##############
        # E-step

        # run the Kalman filter and smoother using the original data 
        # which possibly contains NaN values where data is missing
        Kalman.dk_filter!(kf, Y, wks, x0, Px0, fwdstate, anymissing)
        Kalman.dk_smoother!(kf, Y, wks, fwdstate)

        #############
        # M-step

        if anymissing && impute_missing
            # impute missing values in EY using the smoother output
            em_impute_kalman!(EY, Y, kfd)
        end

        # update model matrices (wks) by maximizing the expected
        # likelihood
        EMstep!(wks, x0, Px0, kfd, EY, M, em_wks, anymissing, use_x0_smooth)

        #############
        # extract new parameters from wks into params vector
        copyto!(old_params, params)
        _update_params!(params, model, wks)
        _update_wks!(wks, model, params)

        #############
        # check for convergence and print progress info
        loglik_new = sum(kfd.loglik)
        dx = mean(abs2, (n - o for (n, o) in zip(params, old_params)))
        df = 2 * abs(loglik - loglik_new) / (abs(loglik) + abs(loglik_new) + eps())

        if verbose == true && mod(iter, 1) == 0
            sl = @sprintf "%.6g" loglik_new
            sx = @sprintf "%.6g" dx
            sf = @sprintf "%.6g" df
            @info "EM iteration $(lpad(iter, 5)): loglik=$sl, df=$sf, dx=$sx"
        end

        loglik = loglik_new

        if df < rftol || dx < axtol
            break
        end

        if loglik_best < loglik
            loglik_best = loglik
            iter_best = iter
            params_best[:] = params
        elseif iter_best + 5 < iter
            # loglik has not improved for 5 iterations.
            if verbose
                @warn "Loglikelihood has not improved for more than 5 interations. Best parameters found at iteration $iter_best"
            end
            params[:] = params_best
            _update_wks!(wks, model, params)
            break
        end
    end

    _scale_model!(wks, Mx, map(inv, Wx), model, em_wks)
    _update_params!(params, model, wks)

    # bring back the original means, if they were given
    for (em_w, onm) in zip(em_wks.observed.loadings, keys(model.observed))
        org_means = getproperty(org_params, onm).mean
        ret_means = getproperty(params, onm).mean
        for (i, v) in enumerate(org_means)
            ret_means[i] += isnan(v) ? Mx[em_w.yinds[i]] : v
        end
    end

    _update_wks!(wks, model, params)

    Y .= Y .* Wx .+ Mx

    return
end

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


