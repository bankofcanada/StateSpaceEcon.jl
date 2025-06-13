##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
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
    em_impute_interpolation!(EY, Y, IT)

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
            j1 = max(first(rows), i - k)
            a1 = max(0, k - i + 1)
            j2 = min(last(rows), i + k)
            a2 = max(0, i + k - last(rows))
            val = (sum(i -> tmp[i], j1:j2) + a1 * tmp[begin] + a2 * tmp[end]) / (2k + 1)
            @inbounds EYj[i] = val
        end
    end
    return EY
end
