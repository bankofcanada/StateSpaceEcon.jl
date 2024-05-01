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

include("em/observed.jl")
include("em/transition.jl")

##################################################################################

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
    use_x0_smooth::Bool=false,
    use_full_XTX::Bool=true,

) where {T}
    @unpack observed, transition = em_wks
    for em_w in observed.loadings
        em_update_observed_block_loading!(wks, kfd, EY, em_w, Val(anymissing), Val(use_full_XTX))
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
    use_full_XTX::Bool=true, # true (default) - use the full XTX matrix when enforcing the loadings constraint, `false` - use XTX assembled according to the missing values pattern in the observed. 
    use_max_norm::Bool=false,
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
        verbose && @info "EM initial guess not given. Calling EMinit!"
        _scale_model!(wks, Mx, Wx, model, em_wks)
        EMinit!(wks, kfd, EY, M, em_wks)
    else
        _update_wks!(wks, model, initial_guess)
        _scale_model!(wks, Mx, Wx, model, em_wks)
        if any(isnan, initial_guess)
            verbose && @info "EM initial guess given but incomplete. Calling EMinit!"
            EMinit!(wks, kfd, EY, M, em_wks)
        else
            verbose && @info "EM initial guess given."
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
        EMstep!(wks, x0, Px0, kfd, EY, M, em_wks, anymissing, use_x0_smooth, use_full_XTX)

        #############
        # extract new parameters from wks into params vector
        copyto!(old_params, params)
        _update_params!(params, model, wks)
        _update_wks!(wks, model, params)

        #############
        # check for convergence and print progress info
        loglik_new = sum(kfd.loglik)
        if use_max_norm
            dx = maximum(abs2, (n - o for (n, o) in zip(params, old_params)))
        else
            dx = mean(abs2, (n - o for (n, o) in zip(params, old_params)))
        end
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

include("em/init.jl")
