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
    mu::MT
    H::HT
    F::FT
    G::GT
    Q::QT
    R::RT
    function KFLinearModel(mu::AbstractVector, H::AbstractMatrix, F::AbstractMatrix,
        G::NoiseShapingMatrix, Q::AbstractMatrix, R::AbstractMatrix)
        ny = length(mu)
        nx = size(H, 2)
        size(H) == (ny, nx) || error("Incompatible size of H")
        size(F) == (nx, nx) || error("Incompatible size of F")
        G isa UniformScaling || size(G) == (nx, nx) || error("Incompatible size of G")
        size(Q) == (ny, ny) || error("Incompatible size of Q")
        size(R) == (nx, nx) || error("Incompatible size of R")
        T = Base.promote_eltype(mu, H, F, G, Q, R)
        eltype(mu) == T || (mu = copyto!(similar(mu, T), mu))
        eltype(H) == T || (H = copyto!(similar(H, T), H))
        eltype(F) == T || (F = copyto!(similar(F, T), F))
        G isa UniformScaling || eltype(G) == T || (G = copyto!(similar(G, T), G))
        eltype(Q) == T || (Q = copyto!(similar(Q, T), Q))
        eltype(R) == T || (R = copyto!(similar(R, T), R))
        new{T,typeof(mu),typeof(H),typeof(F),typeof(G),typeof(Q),typeof(R)}(mu, H, F, G, Q, R)
    end
end

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
    struct KFilter{ET<:Real, NS, NO} 
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
    x::MVector{NS,ET}
    Px::MMatrix{NS,NS,ET}
    x_pred::MVector{NS,ET}
    Px_pred::MMatrix{NS,NS,ET}
    error_y::MVector{NO,ET}
    y_pred::MVector{NO,ET}
    Py_pred::MMatrix{NO,NO,ET}
    Pxy_pred::MMatrix{NS,NO,ET}
    # K::MMatrix{NS, NO, ET}
    x_smooth::MVector{NS,ET}
    Px_smooth::MMatrix{NS,NS,ET}
    Pxx_smooth::MMatrix{NS,NS,ET}
    y_smooth::MVector{NO,ET}
    Py_smooth::MMatrix{NO,NO,ET}
    # J::MMatrix{ET}
    ###### general use storage
    A_xy::MMatrix{NS,NO,ET}
    A_yx::MMatrix{NO,NS,ET}
    A_xx::MMatrix{NS,NS,ET}
    ###### Reference to the data collection we need to fill in
    kfd::Ref{AbstractKFData{ET,RANGE,NS,NO} where {RANGE}}
end

function KFilter(kfd::AbstractKFData{ET,RANGE,NS,NO}) where {ET,RANGE,NS,NO}
    return KFilter{ET,NS,NO}(
        MVector{NS,ET}(undef),
        MMatrix{NS,NS,ET}(undef),
        MVector{NS,ET}(undef),
        MMatrix{NS,NS,ET}(undef),
        MVector{NO,ET}(undef),
        MVector{NO,ET}(undef),
        MMatrix{NO,NO,ET}(undef),
        MMatrix{NS,NO,ET}(undef),
        MVector{NS,ET}(undef),
        MMatrix{NS,NS,ET}(undef),
        MMatrix{NS,NS,ET}(undef),
        MVector{NO,ET}(undef),
        MMatrix{NO,NO,ET}(undef),
        #############
        MMatrix{NS,NO,ET}(undef),
        MMatrix{NO,NS,ET}(undef),
        MMatrix{NS,NS,ET}(undef),
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

#     function KFilter(kfd::AbstractKFData{RANGE,NS,NO,ET}) where {RANGE,NS,NO,ET}
#         range = isa(RANGE, UnitRange) ? RANGE : 1:RANGE
#         x = zeros(ET, NS)
#         Px = zeros(ET, NS, NS)
#         x_pred = zeros(ET, NS)
#         Px_pred = zeros(ET, NS, NS)
#         error_y = zeros(ET, NO)
#         y_pred = zeros(ET, NO)
#         Py_pred = zeros(ET, NO, NO)
#         # Pxy and K occupy the same memory.
#         # We can get away with it because Pxy is only used to compute K
#         K = Pxy_pred = zeros(ET, NS, NO)
#         # smoother things
#         x_smooth = zeros(ET, NS)
#         Px_smooth = zeros(ET, NS, NS)
#         y_smooth = zeros(ET, NO)
#         Py_smooth = zeros(ET, NO, NO)
#         # Pxx and J occupy the same memory.
#         # We can get away with it because Pxx is only used to compute J
#         J = Pxx_smooth = zeros(ET, NS, NS)
#         new{typeof(kfd),ET}(NS, NO, range,
#             x, Px, x_pred, Px_pred,
#             error_y, y_pred, Py_pred, Pxy_pred, K,
#             x_smooth, Px_smooth, Pxx_smooth, y_smooth, Py_smooth, J, kfd)
#     end
# end
