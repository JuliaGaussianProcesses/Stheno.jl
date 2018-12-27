import Base: iszero, zero, *

# Define addition of zeros.
for T in [
    :FiniteZeroKernel,
    :LhsFiniteZeroCrossKernel,
    :RhsFiniteZeroCrossKernel,
    :FiniteZeroCrossKernel,
]
    @eval +(x::$T, x′::$T) = zero(x)
end

# Define zeros for block stuff.
zero(μ::BlockMean) = BlockMean(zero.(μ.μ))
zero(k::BlockCrossKernel) = BlockCrossKernel(zero.(k.ks))
zero(k::BlockKernel) = BlockKernel(zero.(k.ks_diag), zero.(k.ks_off))
