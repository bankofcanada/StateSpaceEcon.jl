##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################


function _smoother_iteration(kf::KFilter, t, model, user_data...)

    kfd = kf.kfd
    
    x_pred = kf.x_pred              # on input contains x_pred[t+1]
    copyto!(x_pred, view(kfd.x_pred, :, t+1))
    Px_pred = kf.Px_pred
    copyto!(Px_pred, view(kfd.Px_pred, :, :, t+1))

    CPx_pred = cholesky!(Symmetric(copy(Px_pred), :U))   # in-place overwrites it
    
    x_smooth = kf.x_smooth          # on input contains x_smooth[t+1]
    Px_smooth = kf.Px_smooth        # on input contains Px_smooth[t+1]
    J = Pxx_pred = kf.Pxx_pred      # input ignored

    y_smooth = kf.y_smooth
    Py_smooth = kf.Py_smooth

    x = view(kfd.x, :, t)           # x and Px are used read-only here, so
    Px = view(kfd.Px, :, :, t)      #    we can take them directly from kfd
    
    # Get Pxx_pred = Eₜ[(xₜ - xₜₜ)(xₜ₊₁ - xₜ₊₁ₜ)']
    #   in the linear case: Pxx_pred[t,t+1] = Px[t] * transposed(F[t])
    kf_predict_x!(t+1, nothing, nothing, Pxx_pred, 
        x, Px, model, user_data...)
    @kfd_set! kfd t Pxx_pred

    # Construct J = Pxx_pred[t,t+1] / Px_pred[t+1]
    # copyto!(J, Pxx_pred)  # no-op since J === Pxx_pred
    rdiv!(J, CPx_pred)

    # Construct smoothed states
    #   -> x_smooth[t] = x[t] + J * (x_smooth[t+1] - x_pred[t+1])
    x_smooth .= x + J * (x_smooth - x_pred)
    # x_pred .-= x_smooth
    # copyto!(x_smooth, x)
    # BLAS.gemv!('N', 1, J, x_pred, 1, x_smooth)

    #   -> Px_smooth[t] = Px[t] + J * (Px_smooth[t+1] - Px_pred[t+1]) * J'
    Px_smooth .= Px + J * (Px_smooth - Px_pred) * transpose(J)

    # Construct smooth observed 
    #   -> y_smooth, Py_smooth ~ regular 
    kf_predict_y!(t, y_smooth, Py_smooth, nothing, 
        x_smooth, Px_smooth, model, user_data...)
    
    # write output
    @kfd_set! kfd t x_smooth Px_smooth y_smooth Py_smooth J

end



function smoother!(kf::KFilter, model, user_data...)
    kfd = kf.kfd
    @unpack x_smooth, Px_smooth, y_smooth, Py_smooth = kf
    t = last(kf.range)
    x_smooth .= kfd_getvalue(kfd, t, Val(:x))
    Px_smooth .= kfd_getvalue(kfd, t, Val(:Px))
    kf_predict_y!(t, y_smooth, Py_smooth, nothing, x_smooth, Px_smooth, model, user_data...)
    @kfd_set! kfd t x_smooth Px_smooth y_smooth Py_smooth
    while t > first(kf.range)
        t = t - 1
        _smoother_iteration(kf, t, model, user_data...)
    end
    return kf
end
