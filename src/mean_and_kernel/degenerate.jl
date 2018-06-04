 """
    DegenerateKernel <: Kernel

A rank-limited kernel, for which `k(x, x′) = kfg(:, x)' * A * kfg(:, x′)`.

# Fields
- `A<:LazyPDMat`:
- `kfg<:CrossKernel`:
"""
struct DegenerateKernel{TA<:LazyPDMat} <: Kernel
    A::TA
    kfg::CrossKernel
    function DegenerateKernel(A::TA, kfg::CrossKernel) where TA<:LazyPDMat
        @assert isfinite(size(kfg, 1)) && size(kfg, 1) == size(A, 1)
        @assert size(A, 1) == size(A, 2)
        return new{TA}(A, kfg)
    end
end
(k::DegenerateKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
(k::DegenerateKernel)(x::AV, x′::AV) =
    binary_obswise(k, reshape(x, length(x), 1), reshape(x′, length(x′), 1))[1]
size(k::DegenerateKernel, N::Int) = size(k.k, 2)

binary_obswise(k::DegenerateKernel, X::AVM) = diag_Xᵀ_A_X(k.A, pairwise(k.kfg, :, X))
function binary_obswise(k::DegenerateKernel, X::AVM, X′::AVM)
    return diag_Xᵀ_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfg, :, X′))
end
pairwise(k::DegenerateKernel, X::AVM) = Xt_A_X(k.A, pairwise(k.kfg, :, X))
function pairwise(k::DegenerateKernel, X::AVM, X′::AVM)
    return Xt_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfg, :, X′))
end

"""
    DegenerateCrossKernel <: CrossKernel

Rank-limited cross-kernel, for which `k(x, x′) = kfg(:, x)' * A * kfh(:, x′)`.
"""
struct DegenerateCrossKernel{TA<:AbstractMatrix} <: CrossKernel
    kfg::CrossKernel
    A::TA
    kfh::CrossKernel
    function DegenerateCrossKernel(kfg::CrossKernel, A::TA, kfh::CrossKernel) where TA<:AM
        @assert isfinite(size(kfg, 1)) && size(kfg, 1) == size(A, 1)
        @assert isfinite(size(kfh, 1)) && size(kfh, 1) == size(A, 2)
        return new{TA}(kfg, A, kfh)
    end
end
(k::DegenerateCrossKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
(k::DegenerateCrossKernel)(x::AV, x′::AV) =
    binary_obswise(k, reshape(x, length(x), 1), reshape(x′, length(x′), 1))[1]
function size(k::DegenerateCrossKernel, N::Int)
    @assert N ∈ (1, 2)
    return N == 1 ? size(k.kfg, 2) : size(k.kfh, 2)
end

function binary_obswise(k::DegenerateCrossKernel, X::AVM, X′::AVM)
    return diag_Xᵀ_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfh, :, X′))
end
function binary_pairwise(k::DegenerateCrossKernel, X::AVM, X′::AVM)
    return Xt_A_Y(pairwise(k.kfg, :, X), k.A, pairwise(k.kfh, :, X′))
end
