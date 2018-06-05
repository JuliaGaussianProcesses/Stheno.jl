"""


"""
struct DeltaSumMean{Tϕ, Tg, TZ} <: MeanFunction
    ϕ::Tϕ
    μ::MeanFunction
    Z::TZ
    g::Tg
end
(μ::DeltaSumMean)(x) = pairwise(μ.ϕ, x, μ.Z)' * unary_obswise(μ.μ, μ.Z) + μ.g(x)
function unary_obswise(μ::DeltaSumMean, X::AVM)
    return pairwise(μ.ϕ, μ.Z, X)' * unary_obswise(μ.μ, μ.Z) + unary_obswise(μ.g, X)
end

"""


"""
struct DeltaSumKernel{Tϕ, TZ} <: Kernel
    ϕ::Tϕ
    k::Kernel
    Z::TZ
end
(k::DeltaSumKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
function (k::DeltaSumKernel)(x::AV, x′::AV)
    return binary_obswise(k, reshape(x, :, 1), reshape(x′, :, 1))[1]
end
size(k::DeltaSumKernel, N::Int) = size(k.ϕ, 2)

function binary_obswise(k::DeltaSumKernel, X::AVM)
    return diag_Xᵀ_A_X(pairwise(k.k, k.Z), pairwise(k.ϕ, k.Z, X))
end
function binary_obswise(k::DeltaSumKernel, X::AVM, X′::AVM)
    return diag_Xᵀ_A_Y(pairwise(k.ϕ, k.Z, X), pairwise(k.k, k.Z), pairwise(k.ϕ, k.Z, X′))
end
function pairwise(k::DeltaSumKernel, X::AVM)
    return Xt_A_X(pairwise(k.k, k.Z), pairwise(k.ϕ, k.Z, X))
end
function pairwise(k::DeltaSumKernel, X::AVM, X′::AVM)
    return Xt_A_Y(pairwise(k.ϕ, k.Z, X), pairwise(k.k, k.Z), pairwise(k.ϕ, k.Z, X′))
end

"""

"""
struct LhsDeltaSumCrossKernel{Tϕ, TZ} <: CrossKernel
    ϕ::Tϕ
    k::CrossKernel
    Z::TZ
end
(k::LhsDeltaSumCrossKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
function (k::LhsDeltaSumCrossKernel)(x::AV, x′::AV)
    return binary_obswise(k, reshape(x, : ,1), reshape(x′, :, 1))[1]
end
size(k::LhsDeltaSumCrossKernel, N::Int) = N == 1 ? size(k.ϕ, 2) : size(k.k, 2)

function binary_obswise(k::LhsDeltaSumCrossKernel, X::AVM, X′::AVM)
    return diag_AᵀB(pairwise(k.ϕ, k.Z, X), pairwise(k.k, k.Z, X′))
end
function pairwise(k::LhsDeltaSumCrossKernel, X::AVM, X′::AVM)
    return pairwise(k.ϕ, k.Z, X)' * pairwise(k.k, k.Z, X′)
end

"""

"""
struct RhsDeltaSumCrossKernel{TZ, Tϕ} <: CrossKernel
    k::CrossKernel
    Z::TZ
    ϕ::Tϕ
end
(k::RhsDeltaSumCrossKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
function (k::RhsDeltaSumCrossKernel)(x::AV, x′::AV)
    return binary_obswise(k, reshape(x, :, 1), reshape(x′, :, 1))[1]
end
size(k::RhsDeltaSumCrossKernel, N::Int) = N == 1 ? size(k.k, 2) : size(k.ϕ, 2)

function binary_obswise(k::RhsDeltaSumCrossKernel, X::AVM, X′::AVM)
    return diag_AᵀB(pairwise(k.k, k.Z, X), pairwise(k.ϕ, k.Z, X′))
end
function pairwise(k::RhsDeltaSumCrossKernel, X::AVM, X′::AVM)
    return pairwise(k.k, k.Z, X)' * pairwise(k.ϕ, k.Z, X′)
end


# """
#     DegenerateKernel <: Kernel

# A rank-limited kernel, for which `k(x, x′) = kfg(:, x)' * A * kfg(:, x′)`.

# # Fields
# - `A<:LazyPDMat`:
# - `kfg<:CrossKernel`:
# """
# struct DegenerateKernel{TA<:LazyPDMat} <: Kernel
#     A::TA
#     kfg::CrossKernel
#     function DegenerateKernel(A::TA, kfg::CrossKernel) where TA<:LazyPDMat
#         @assert isfinite(size(kfg, 1)) && size(kfg, 1) == size(A, 1)
#         @assert size(A, 1) == size(A, 2)
#         return new{TA}(A, kfg)
#     end
# end
# (k::DegenerateKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
# (k::DegenerateKernel)(x::AV, x′::AV) =
#     binary_obswise(k, reshape(x, length(x), 1), reshape(x′, length(x′), 1))[1]
# size(k::DegenerateKernel, N::Int) = size(k.k, 2)

# binary_obswise(k::DegenerateKernel, X::AVM) = diag_Xᵀ_A_X(k.A, pairwise(k.kfg, :, X))
# function binary_obswise(k::DegenerateKernel, X::AVM, X′::AVM)
#     return diag_Xᵀ_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfg, :, X′))
# end
# pairwise(k::DegenerateKernel, X::AVM) = Xt_A_X(k.A, pairwise(k.kfg, :, X))
# function pairwise(k::DegenerateKernel, X::AVM, X′::AVM)
#     return Xt_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfg, :, X′))
# end

# """
#     DegenerateCrossKernel <: CrossKernel

# Rank-limited cross-kernel, for which `k(x, x′) = kfg(:, x)' * A * kfh(:, x′)`.
# """
# struct DegenerateCrossKernel{TA<:AbstractMatrix} <: CrossKernel
#     kfg::CrossKernel
#     A::TA
#     kfh::CrossKernel
#     function DegenerateCrossKernel(kfg::CrossKernel, A::TA, kfh::CrossKernel) where TA<:AM
#         @assert isfinite(size(kfg, 1)) && size(kfg, 1) == size(A, 1)
#         @assert isfinite(size(kfh, 1)) && size(kfh, 1) == size(A, 2)
#         return new{TA}(kfg, A, kfh)
#     end
# end
# (k::DegenerateCrossKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
# (k::DegenerateCrossKernel)(x::AV, x′::AV) =
#     binary_obswise(k, reshape(x, length(x), 1), reshape(x′, length(x′), 1))[1]
# function size(k::DegenerateCrossKernel, N::Int)
#     @assert N ∈ (1, 2)
#     return N == 1 ? size(k.kfg, 2) : size(k.kfh, 2)
# end

# function binary_obswise(k::DegenerateCrossKernel, X::AVM, X′::AVM)
#     return diag_Xᵀ_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfh, :, X′))
# end
# function binary_pairwise(k::DegenerateCrossKernel, X::AVM, X′::AVM)
#     return Xt_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfh, :, X′))
# end
