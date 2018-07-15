import Base: iszero, zero, *

const ZM = ZeroMean{Float64}
const ZK = ZeroKernel{Float64}

# Define the zero element.

zero(μ::MeanFunction) = length(μ) < Inf ? FiniteZeroMean(eachindex(μ)) : ZM()
zero(μ::BlockMean) = BlockMean(zero.(μ.μ))

function zero(k::CrossKernel)
    if size(k, 1) < Inf && size(k, 2) < Inf
        return FiniteZeroCrossKernel(eachindex(k, 1), eachindex(k, 2))
    elseif size(k, 1) < Inf
        return LhsFiniteZeroCrossKernel(eachindex(k, 1))
    elseif size(k, 2) < Inf
        return RhsFiniteZeroCrossKernel(eachindex(k, 2))
    else
        return ZeroKernel{Float64}()
    end
end
zero(k::BlockCrossKernel) = BlockCrossKernel(zero.(k.ks))

zero(k::Kernel) = length(k) < Inf ? FiniteZeroKernel(eachindex(k)) : ZK()
@noinline zero(k::BlockKernel) = BlockKernel(zero.(k.ks_diag), zero.(k.ks_off))


# Define addition of zeros.
for T in [:ZeroMean, :FiniteZeroMean, :ZeroKernel, :FiniteZeroKernel,
        :LhsFiniteZeroCrossKernel, :RhsFiniteZeroCrossKernel, :FiniteZeroCrossKernel]
    @eval +(x::$T, x′::$T) = zero(x)
end

function +(μ::MeanFunction, μ′::MeanFunction)
    @assert size(μ) == size(μ′)
    if iszero(μ)
        return μ′
    elseif iszero(μ′)
        return μ
    else
        return CompositeMean(+, μ, μ′)
    end
end
function +(k::CrossKernel, k′::CrossKernel)
    @assert size(k) == size(k′)
    if iszero(k)
        return k′
    elseif iszero(k′)
        return k
    else
        return CompositeCrossKernel(+, k, k′)
    end
end
function +(k::Kernel, k′::Kernel)
    @assert size(k) == size(k′)
    if iszero(k)
        return k′
    elseif iszero(k′)
        return k
    else
        return CompositeKernel(+, k, k′)
    end
end

function *(μ::MeanFunction, μ′::MeanFunction)
    @assert size(μ) == size(μ′)
    return iszero(μ) || iszero(μ′) ? zero(μ) : CompositeMean(*, μ, μ′)
end
function *(k::CrossKernel, k′::CrossKernel)
    @assert size(k) == size(k′)
    return iszero(k) || iszero(k′) ? zero(k) : CompositeCrossKernel(*, k, k′)
end
function *(k::Kernel, k′::Kernel)
    @assert size(k) == size(k′)
    return iszero(k) || iszero(k′) ? zero(k) : CompositeKernel(*, k, k′)
end
