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
