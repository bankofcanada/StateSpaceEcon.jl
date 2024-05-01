##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
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
        if use_x0_smooth
            # XTX_12 is already corrected above
            # indices for EXp and PXp are shifted by 1
            EXp = transpose(@view x_smooth[xinds_1, begin:end])
            PXp = @view(Px_smooth[xinds_1, xinds_1, begin:end])
        else
            EXp = EXp_1
            PXp = @view(Px_smooth[xinds_1, xinds_1, begin+1:end])
        end
        _em_update_transition_covar!(new_Q, EXp, PXp, new_A, XTX_12)
        copyto!(vQ, new_Q)
    end

    return new_A, new_Q
end

##################################################################################

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
    Qd = Q.data
    mul!(Qd, transpose(EX), EX)
    sum!(Qd, PX, init=false)
    mul!(Qd, A, transpose(PXpm), -1, 1)
    ldiv!(NT, Qd)
    return Q
end

##################################################################################


struct EM_Transition_Wks{T,TBLK}
    blocks::TBLK
end
function em_transition_wks(M::DFM, wks::DFMKalmanWks{T}) where {T}
    @unpack model, params = M
    blocks = (; (nm => em_transition_block_wks(nm, M, wks) for nm in keys(model.components))...)
    return EM_Transition_Wks{T,typeof(blocks)}(blocks)
end


