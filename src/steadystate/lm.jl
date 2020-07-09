

"""
Levenberg–Marquardt step of the steady state solver.
"""

# This line intentionally left blank.

"""
LMData

A data structure that holds the necessary buffers and internal data for
performing a step of the Levenberg–Marquardt algorithm

!!! warning
    Internal use. Do not call directly
"""
struct LMData
    sd::SolverData
    M::AbstractArray{Float64,2}
    Jd::AbstractArray{Float64,1}
    JtR::AbstractArray{Float64,1}
    v_buffer::AbstractArray{Float64,1}
    r_buffer::AbstractArray{Float64,1}
    params::AbstractArray{Float64,1}
end

"""
    LMData(model)
    LMData(model, sd::SolverData)

Make an instance of the LMData for the given model. If solver data is not also given,
then it is used, instead of creating a default SolverData for the model.
"""
LMData(model::Model) = LMData(model, SolverData(model))
LMData(model::Model, sd::SolverData) = LMData(sd, 
                Array{Float64}(undef, sd.nvars, sd.nvars), 
                Array{Float64}(undef, sd.nvars),
                Array{Float64}(undef, sd.nvars),
                Array{Float64}(undef, sd.nvars),
                Array{Float64}(undef, sd.neqns),
                [0.1, 10.]
        )


"""
    step_nr!(x, dx, resid, J, lm::LMData; verbose=false)

Attempt a Levenberg–Marquardt step. 
The `lm` structure and the `dx` vector would be updated.
Vectors `x`, `resid` and the matrix `J` are read-only inputs.

!!! warning
    Internal function, do not call directly.
"""
function step_lm!(xx::AbstractArray{Float64,1}, dx::AbstractArray{Float64,1},
                  R::AbstractArray{Float64,1}, J::AbstractArray{Float64,2},
                  lm::LMData; verbose::Bool = false)
    lambda, nu = lm.params
    nx = length(dx)
    lm.JtR .= J' * R
    lm.M .= J' * J
    lm.Jd[:] = sum(abs2, J; dims = 1)
    lm.Jd .= max.(lm.Jd, 1e-10)  # make sure no zeros
    n2R = sum(abs2, R)
    for i = 1:nx
        @inbounds lm.M[i,i] += lambda * lm.Jd[i]
    end
    dx .= lm.M \ lm.JtR
    # predicted residual squared norm
    n2PR = sum(abs2, R .- J * dx)
    # actual residual and its squared norm
    lm.v_buffer .= xx .- dx
    # reusing JtR for the actual residual
    global_SS_R!(lm.r_buffer, lm.v_buffer, lm.sd)
    n2AR = sum(abs2, lm.r_buffer)
    # quality of step
    qual = (n2AR - n2R) / (n2PR - n2R)
    # adjust lambda
    if qual > 0.75
        # good quality, extend trust region
        lambda = max(1e-16, lambda / nu)
    elseif qual < 1e-3
        # bad quality, shrink trust region
        lambda = min(1e16, lambda * nu)
    end
    lm.params[:] = [lambda, nu]
    return nothing
end
@assert precompile(step_lm!, (Array{Float64,1}, Array{Float64,1}, Array{Float64,1}, Array{Float64,2}, LMData))

"""
    first_step_nr!(x, dx, resid, J, lm::LMData; verbose=false)

Make the first step of a Levenberg–Marquardt algorithm. Involves determining the initial trust region.
The `lm` structure and the `dx` vector would be updated.
Vectors `x`, `resid` and the matrix `J` are read-only inputs.

!!! warning
    Internal function, do not call directly.
"""
function first_step_lm!(xx::AbstractArray{Float64,1}, dx::AbstractArray{Float64,1},
                R::AbstractArray{Float64,1}, J::AbstractArray{Float64,2},
                lm::LMData; verbose = false,
)
    lambda, nu = lm.params
    nx = length(dx)
    lm.JtR .= J' * R
    lm.M .= J' * J
    lm.Jd[:] = sum(abs2, J; dims = 1)
    lm.Jd .= max.(lm.Jd, 1e-10)  # make sure no zeros
    n2R = sum(abs2, R)
    qual = 0.0
    coef = 1.0
    n2AR = -1.0
    while qual < 1e-3
        for i = 1:nx
            @inbounds lm.M[i,i] += lambda * coef * lm.Jd[i]
            # the above should be diag(M) += lambda * Jd
            # we have coef = (1-1/nu) for the iterations after the first one
            # in order to subtract the previous lambda from diag(M)
            # each iteration we shrink the trust region by setting lambda *= nu
        end
        dx .= lm.M \ lm.JtR
        # predicted residual norm
        n2PR = sum(abs2, R .- J * dx)
        # actual residual and its norm
        lm.v_buffer .= xx .- dx
        qual = 0.0
        try
            global_SS_R!(lm.r_buffer, lm.v_buffer, lm.sd)
            n2AR = sum(abs2, lm.r_buffer)
            # quality of step
            qual = (n2AR - n2R) / (n2PR - n2R)
        catch
        end
        if qual > 0.75 # very good quality, extend the trust region
            lambda = lambda / nu
            break
        end
        if lambda >= 1e16 # lambda is getting too large
            lambda = 1e16
            break
        end
        # shrink the trust regions
        lambda = lambda * nu
        coef = 1.0 - 1.0 / nu
    end
    lm.params[:] = [lambda, nu]
    nothing
end
@assert precompile(first_step_lm!, (Array{Float64,1}, Array{Float64,1}, Array{Float64,1}, Array{Float64,2}, LMData))

