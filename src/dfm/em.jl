##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

_my_cholesky!(X, ::Val{:brave}) = cholesky!(Symmetric(X))
_my_cholesky!(X, ::Val{:verbose}) = begin
    try
        cholesky!(Symmetric(X))
    catch
        ev = eigen(X).values
        @error "Cholesky failed at "
        println("X=", X)
        println("ev=", ev)
        rethrow()
    end
end
_my_cholesky!(X) = _my_cholesky!(X, Val(:verbose))


include("em/constraints.jl")
include("em/interpolation.jl")
include("em/observed.jl")
include("em/transition.jl")

##################################################################################

struct EM_Wks{T,TOBS,TTRS}
    observed::TOBS
    transition::TTRS
end

function em_workspace(M::DFM{T}, LM::KFLinearModel{T}; orthogonal_factors_ics=true) where {T}
    obs = em_observed_wks(M, LM; orthogonal_factors_ics)
    trans = em_transition_wks(M, LM)
    return EM_Wks{T,typeof(obs),typeof(trans)}(obs, trans)
end


##################################################################################



function EMstep!(LM::KFLinearModel{T}, x0::AbstractVector{T}, Px0::AbstractMatrix{T},
    kfd::Kalman.AbstractKFData{T}, EY::AbstractMatrix{T}, M::DFM, em_wks::EM_Wks{T},
    anymissing::Bool, # =any(isnan, EY)
    use_x0_smooth::Bool=false,
    use_full_XTX::Bool=true,) where {T}
    @unpack observed, transition = em_wks
    for em_w in observed.loadings
        em_update_observed_block_loading!(LM, kfd, EY, em_w, Val(anymissing), Val(use_full_XTX))
    end
    em_update_observed_covar!(LM, kfd, EY, observed.covars, Val(anymissing))
    for em_w in transition.blocks
        em_update_transition_block!(LM, kfd, EY, em_w, Val(use_x0_smooth))
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
    return LM
end



##################################################################################


function em_scale_model!(LM::KFLinearModel{T}, Mx, Wx, model::DFMModel, em_wks::EM_Wks) where {T}
    # IDEA: factors (CommonComponents) have loadings, so we scale the loadings
    #       obs noise and idiosyncratic components figure in with fixed loadings,
    #       so we scale their noises covariances instead.
    # scale the loadings of factors
    μ, Λ, R, Q = LM.mu, LM.H, LM.Q, LM.R
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
        constraint === nothing && continue
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
    return LM
end

"""
    EMestimate!(M::DFM, Y, <args>; <options>)

Main function to run DFM estimation by the EM algorithm. The DFM model instance
will be updated in place and returned.

On input `M.params` must contain `NaN` for parameters that are to be estimated
and numbers for parameters that are known and will not be estimated. On output
`M.params` will contain the estimated parameters, that is non-NaN values will be
preserved while NaN values will be overwritten with their estimated values.

Observed data `Y` may contain an arbitrary pattern of missing values.

Additional positional arguments are optional. Their order is as follows:
 * `x0` - state vector at time 0 used for the Kalman Filter. Default is the zero
   vector.
 * 'Px0` - state covariance matrix at time 0 used for the Kalman Filter. Default
   is 1e-10*I
 * `LM` - an instance of `KFLinearModel` for the same model as `M`. Default
   value is `kf_linear_model(M)`
 * `em_wks` - an instance of `EM_Wks` to be used by the EM estimator. Default
   value is `em_workspace(M, LM)`
 * `kfd` - an instance of `AbstractKFData` to be used by the EM estimator. The
   provided instance must be suitable for running the Kalman Smoother. Default
   value is `KFDataSmoother(size(Y,1), M, Y)`
 * `kf` - an instance of `KFilter` to be used by the EM estimator. The default
   value is `KFilter(kfd)`

Named options
 * `verbose`
 * You can save yourself a few CPU cycles by setting `anymissing` to `true` or
   `false`, if known. Otherwise the default is computed by checking `Y`.
 * `fwdstate` controls whether the Kalman Filter starts with states at time 0 or
   at time 1. Default value is `false`, which means time 0. See also
   [`kf_filter`](@ref) and [`kf_smoother`](@ref) and note the the default here
   is different that in the `Kalman` sub-module.
 * `maxiter`, `rftol`, and `axtol` control the convergence/termination checks of
   the EM estimation iteration.
 * `initial_guess` can be used to provide an initial guess for `M.params`. If
   not given (`nothing`), or if the given vector has any missing values, then an
   internal procedure is called to generate the initial guess. Default is
   `nothing`. If given and all values are non-zero, they will be used as initial
   guess. *N.B. it is the responsibility of the caller to ensure that all non-NaN
   values in `M.params` and `initial_guess` match; this is neither verified nor
   enforced here.*
 * `impute_missing` controls how to deal with missing values in `Y`. If set to
   `false` (default) then the missing values are treated by a modification of
   the EM algorithm formulas as in Banbura & Modugno 2014. Otherwise, missing
   data are filled in using their expected values (from the Kalman smoother) on
   each iteration of the EM estimation.
 * `use_max_norm` as opposed to root mean squared difference. Default is
   `false`.
 * `strict` is set to `true` by default. If the requested tolerance is not
   reached after `maxiter` iterations, this results in an error. Set to `false`
   to simply return the best result so far.

"""
function EMestimate!(M::DFM{T}, Y::AbstractMatrix,
    x0::AbstractVector{T}=zeros(kf_length_x(M)),
    Px0::AbstractMatrix{T}=Matrix{T}(1e-10 * I(kf_length_x(M))),
    LM::KFLinearModel{T}=kf_linear_model(M),
    em_wks::EM_Wks{T}=em_workspace(M, LM),
    kfd::Kalman.AbstractKFData{T}=KFDataSmoother(T, size(Y, 1), M, Y),
    kf::KFilter{T}=KFilter(kfd)
    ;
    fwdstate::Bool=false,
    initial_guess::Union{Nothing,AbstractVector}=nothing,
    maxiter=150, rftol=1e-6, axtol=1e-6,
    verbose=false,
    impute_missing::Bool=false,  # true - use Kalman smoother to impute missing data, false - treat missing data as in Banbura & Modugno 2014
    use_x0_smooth::Bool=false, # true - use x0_smooth in the EMstep update (experimental). Set to `false` for normal operation.
    use_full_XTX::Bool=true, # true (default) - use the full XTX matrix when enforcing the loadings constraint, `false` - use XTX assembled according to the missing values pattern in the observed. The results are not the same, but should be close
    use_max_norm::Bool=false,
    anymissing::Bool=any(isnan, Y),
    strict::Bool=true,
) where {T}
    @unpack model, params = M

    if !any(isnan, params)
        @error "No parameters have been marked NaN for estimation."
        return true
    end

    org_params = copy(params)
    old_params = copy(params)

    # scale data
    Mx = nanmean(Y, dims=1)
    for i = eachindex(Mx)
        v = LM.mu[i]
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
        em_scale_model!(LM, Mx, Wx, model, em_wks)
        EMinit!(LM, kfd, EY, M, em_wks)
    else
        update_dfm_lm!(LM, model, initial_guess)
        em_scale_model!(LM, Mx, Wx, model, em_wks)
        if any(isnan, initial_guess)
            verbose && @info "EM initial guess given but incomplete. Calling EMinit!"
            EMinit!(LM, kfd, EY, M, em_wks)
        else
            verbose && @info "EM initial guess given."
        end
    end
    update_dfm_params!(params, model, LM)

    if anymissing && !impute_missing
        copyto!(EY, Y)
    end

    loglik = -Inf
    loglik_best = -Inf
    iter_best = 0
    params_best = copy(params)
    iter = 0
    while iter < maxiter
        iter += 1

        ##############
        # E-step

        # run the Kalman filter and smoother using the original data,
        # which possibly contains NaN values where data is missing
        kf_filter!(kf, Y, x0, Px0, LM; fwdstate, anymissing)
        kf_smoother!(kf, LM; fwdstate)

        #############
        # M-step

        if anymissing && impute_missing
            # impute missing values in EY using the smoother output
            em_impute_kalman!(EY, Y, kfd)
        end

        # update model matrices (LM) by maximizing the expected likelihood
        EMstep!(LM, x0, Px0, kfd, EY, M, em_wks, anymissing, use_x0_smooth, use_full_XTX)

        #############
        # extract new parameters from wks into params vector
        copyto!(old_params, params)
        update_dfm_params!(params, model, LM)
        update_dfm_lm!(LM, model, params)

        #############
        # check for convergence and print progress info
        loglik_new = sum(kfd.loglik)
        if use_max_norm
            dx = maximum(abs, (n - o for (n, o) in zip(params, old_params)))
        else
            dx = √mean(abs2, (n - o for (n, o) in zip(params, old_params)))
        end
        df = 2 * abs(loglik - loglik_new) / (abs(loglik) + abs(loglik_new) + eps())

        if verbose && mod(iter, 1) == 0
            sl = @sprintf "%.6g" loglik_new
            sx = @sprintf "%.6g" dx
            sf = @sprintf "%.6g" df
            @info "EM iteration $(lpad(iter, 5)): loglik=$sl, df=$sf, dx=$sx"
        end

        loglik = loglik_new

        if df < rftol || dx < axtol
            if verbose
                @info "EM reached desired tolerance: df = $df < $rftol or dx = $dx < $axtol"
            end
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
            update_dfm_lm!(LM, model, params)
            break
        end
    end

    if iter >= maxiter
        if strict
            error("EM failed to converge after `maxiter` iterations.")
        end
        if verbose 
            @info "EM reached maximum iterations: iter = $iter >= $maxiter"
        end
    end

    em_scale_model!(LM, Mx, map(inv, Wx), model, em_wks)
    update_dfm_params!(params, model, LM)

    # bring back the original means, if they were given
    for (em_w, onm) in zip(em_wks.observed.loadings, keys(model.observed))
        org_means = getproperty(org_params, onm).mean
        ret_means = getproperty(params, onm).mean
        for (i, v) in enumerate(org_means)
            ret_means[i] += isnan(v) ? Mx[em_w.yinds[i]] : v
        end
    end

    update_dfm_lm!(LM, model, params)
    Y .= Y .* Wx .+ Mx
    kf_filter!(kf, Y, x0, Px0, LM; fwdstate, anymissing)
    kf_smoother!(kf, LM; fwdstate)
    
    return iter < maxiter 
end

include("em/init.jl")
