##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

###############################################################################
#    KFLinearModel
###############################################################################

const NoiseShapingMatrix{T} = Union{UniformScaling{T},AbstractMatrix{T}}

"""
    struct KFLinearModel ... end

A simple data structure that can be used as an interface from the linear state
space model to the Kalman filter functionality provided by this module,
$(@__MODULE__).
"""
struct KFLinearModel{T<:Real,MT<:AbstractVector{T},HT<:AbstractMatrix{T},
    FT<:AbstractMatrix{T},GT<:NoiseShapingMatrix{T},
    QT<:AbstractMatrix{T},RT<:AbstractMatrix{T}}
    mu::MT          # vector of observed means
    H::HT           # loadings matrix
    F::FT           # transition matrix
    G::GT           # noise shaping matrix
    Q::QT           # observed noise covariance
    R::RT           # states noise covariance
    function KFLinearModel(T::Type{<:Real}, mu::AbstractVector, H::AbstractMatrix, F::AbstractMatrix,
        G::NoiseShapingMatrix, Q::AbstractMatrix, R::AbstractMatrix)
        ny = length(mu)
        nx = size(H, 2)
        size(H) == (ny, nx) || error("Incompatible size of H")
        size(F) == (nx, nx) || error("Incompatible size of F")
        G isa UniformScaling || size(G) == (nx, nx) || error("Incompatible size of G")
        size(Q) == (ny, ny) || error("Incompatible size of Q")
        size(R) == (nx, nx) || error("Incompatible size of R")
        eltype(mu) == T || (mu = copyto!(similar(mu, T), mu))
        eltype(H) == T || (H = copyto!(similar(H, T), H))
        eltype(F) == T || (F = copyto!(similar(F, T), F))
        G isa UniformScaling || eltype(G) == T || (G = copyto!(similar(G, T), G))
        eltype(Q) == T || (Q = copyto!(similar(Q, T), Q))
        eltype(R) == T || (R = copyto!(similar(R, T), R))
        new{T,typeof(mu),typeof(H),typeof(F),typeof(G),typeof(Q),typeof(R)}(mu, H, F, G, Q, R)
    end
end

KFLinearModel(mu::AbstractVector, H::AbstractMatrix, F::AbstractMatrix,
    G::NoiseShapingMatrix, Q::AbstractMatrix, R::AbstractMatrix) =
    KFLinearModel(Base.promote_eltype(mu, H, F, G, Q, R), mu, H, F, G, Q, R)

KFLinearModel(model, user_data...) = KFLinearModel(Float64, model, user_data...)
function KFLinearModel(T::Type{<:Real}, model, user_data...)
    kf_is_linear(model, user_data...) || error("Not a linear model.")
    nx = kf_length_x(model, user_data...)
    ny = kf_length_y(model, user_data...)
    if kf_state_noise_shaping(model, user_data...)
        G = Matrix{T}(undef, nx, nx)
    else
        G = one(T) * I
    end
    return KFLinearModel(Vector{T}(undef, ny), Matrix{T}(undef, ny, nx),
        Matrix{T}(undef, nx, nx), G,
        Matrix{T}(undef, ny, ny), Matrix{T}(undef, nx, nx)
    )
end

# implement the API for KFLinearModel
kf_is_linear(::KFLinearModel) = true
kf_length_x(m::KFLinearModel) = size(m.F, 1)
kf_length_y(m::KFLinearModel) = size(m.H, 1)
kf_linear_model(m::KFLinearModel) = deepcopy(m)
kf_state_noise_shaping(m::KFLinearModel) = m.G isa AbstractMatrix

###############################################################################
#    KFLinearModel
###############################################################################

"""
    struct KFilter{ET<:Real,NS,NO} 
        ... 
        kfd
    end
    kf = KFilter(kdf::AbstractKFData)

A data structure that provides the workspace needed when running the Kalman
Filter. It contains pre-allocated arrays with sizes corresponding to the number
of states (`NS`) and the number of observed variables (`NO`).

It can be constructed from a given Kalman data container. A reference to the
data container is stored and can be swapped with another one, as long as it has
the same number of state and observed variables. This allows the allocated
storage to be reused in multiple runs.

"""
struct KFilter{ET<:Real,NS,NO}
    # nx::Int
    # ny::Int
    x::Vector{ET}
    Px::Matrix{ET}
    x_pred::Vector{ET}
    Px_pred::Matrix{ET}
    error_y::Vector{ET}
    y_pred::Vector{ET}
    Py_pred::Matrix{ET}
    Pxy_pred::Matrix{ET}
    # K::Matrix{ET}
    x_smooth::Vector{ET}
    Px_smooth::Matrix{ET}
    Pxx_smooth::Matrix{ET}
    y_smooth::Vector{ET}
    Py_smooth::Matrix{ET}
    # J::Matrix{ET}
    ###### general use storage
    A_x
    A_xy::Matrix{ET}
    A_yx::Matrix{ET}
    A_xx::Matrix{ET}
    B_xx::Matrix{ET}
    ###### Reference to the data collection we need to fill in
    kfd::Ref{AbstractKFData{ET,RANGE,NS,NO} where {RANGE}}
end

function KFilter(kfd::AbstractKFData{ET,RANGE,NS,NO}) where {ET,RANGE,NS,NO}
    return KFilter{ET,NS,NO}(
        Vector{ET}(undef,NS),
        Matrix{ET}(undef,NS,NS),
        Vector{ET}(undef,NS),
        Matrix{ET}(undef,NS,NS),
        Vector{ET}(undef,NO),
        Vector{ET}(undef,NO),
        Matrix{ET}(undef,NO,NO),
        Matrix{ET}(undef,NS,NO),
        Vector{ET}(undef,NS),
        Matrix{ET}(undef,NS,NS),
        Matrix{ET}(undef,NS,NS),
        Vector{ET}(undef,NO),
        Matrix{ET}(undef,NO,NO),
        #############
        Vector{ET}(undef,NS),
        Matrix{ET}(undef,NS,NO),
        Matrix{ET}(undef,NO,NS),
        Matrix{ET}(undef,NS,NS),
        Matrix{ET}(undef,NS,NS),
        kfd
    )
end

Base.eltype(::KFilter{ET}) where {ET} = ET
function Base.getproperty(kf::KFilter, name::Symbol)
    if name == :J
        name = Pxx_smooth   # J and Pxx_smooth occupy the same memory because Pxx_pred is only used to compute J
    end
    if name == :K
        name = :Pxy_pred    # K abd Pxy_pred occupy the same memory because Pxy_pred is only used to compute K
    end
    if name == :kfd
        # unpack the Ref
        return Base.getfield(kf, name).x
    else
        return Base.getfield(kf, name)
    end
end

kf_length_x(::KFilter{ET,NS,NO}) where {ET,NS,NO} = NS
kf_length_y(::KFilter{ET,NS,NO}) where {ET,NS,NO} = NO
kf_time_periods(kf::KFilter) = kf_time_periods(kf.kfd)


function Base.setproperty!(kf::KFilter, name::Symbol, val)
    if name == :kfd && val isa AbstractKFData
        # we're allowed to change the data reference. 
        setindex!(getfield(kf, :kfd), val)
    else
        # the rest is immutable, so let Julia handle the error message
        setfield!(kf, name, val)
    end
end

