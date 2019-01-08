import Base: +, *
export finite, lhs_finite, rhs_finite


########################
# MeanFunction algebra #
########################

# Adding things to mean functions.
+(α::Real, m::MeanFunction) = UnaryMean(x->α + x, m)
+(m::MeanFunction, α::Real) = UnaryMean(x->x + α, m)
+(m::MeanFunction, m′::MeanFunction) = BinaryMean(+, m, m′)

# Optimise addition for zero.
+(m::ZeroMean, m′::ZeroMean) = m
+(m::ZeroMean, m′::MeanFunction) = m′
+(m::MeanFunction, m′::ZeroMean) = m

# Multiplying mean functions by things.
*(α::Real, m::MeanFunction) = UnaryMean(x->α * x, m)
*(m::MeanFunction, α::Real) = UnaryMean(x->x * α, m)
*(m::MeanFunction, m′::MeanFunction) = BinaryMean(*, m, m′)

# Optimise multiplication for zero.
*(m::ZeroMean, m′::ZeroMean) = m
*(m::ZeroMean, m′::MeanFunction) = m
*(m::MeanFunction, m′::ZeroMean) = m′

# Creating a finite-dimensional mean function.
finite(m::MeanFunction, x::AV) = FiniteMean(m, x)


##################################
# Kernel and CrossKernel algebra #
##################################

# Adding things to (Cross)Kernels.
+(α::Real, k::CrossKernel) = BinaryCrossKernel(+, ConstKernel(α), k)
+(k::CrossKernel, α::Real) = BinaryCrossKernel(+, k, ConstKernel(α))
+(α::Real, k::Kernel) = BinaryKernel(+, ConstKernel(α), k)
+(k::Kernel, α::Real) = BinaryKernel(+, k, ConstKernel(α))
+(k::CrossKernel, k′::CrossKernel) = BinaryCrossKernel(+, k, k′)
+(k::Kernel, k′::Kernel) = BinaryKernel(+, k, k′)

# Optimise addition with zero.
+(k::ZeroKernel, k′::ZeroKernel) = k
+(k::ZeroKernel, k′::CrossKernel) = k′
+(k::CrossKernel, k′::ZeroKernel) = k
+(k::ZeroKernel, k′::Kernel) = k′
+(k::Kernel, k′::ZeroKernel) = k

# Optimise for addition of `ConstKernel`s.
+(k::ConstKernel, k′::ConstKernel) = ConstKernel(k.c + k′.c)

# Multiply kernels by things.
*(α::Real, k::CrossKernel) = LhsCross(x->α, k)
*(k::CrossKernel, α::Real) = RhsCross(k, x->α)
*(k::CrossKernel, k′::CrossKernel) = BinaryCrossKernel(*, k, k′)
*(k::Kernel, k′::Kernel) = BinaryKernel(*, k, k′)

# Optimise multiplication by zeros.
*(k::ZeroKernel, k′::ZeroKernel) = k
*(k::ZeroKernel, k′::CrossKernel) = k
*(k::CrossKernel, k′::ZeroKernel) = k′
*(k::ZeroKernel, k′::Kernel) = k
*(k::Kernel, k′::ZeroKernel) = k′

# Optimise multiplication by one.
*(k::OneKernel, k′::OneKernel) = k
*(k::OneKernel, k′::CrossKernel) = k′
*(k::CrossKernel, k′::OneKernel) = k
*(k::OneKernel, k′::Kernel) = k′
*(k::Kernel, k′::OneKernel) = k

# Optimise for product of `ConstKernel`s.
*(k::ConstKernel, k′::ConstKernel) = ConstKernel(k.c * k′.c)

# Create finite-dimensional kernels.
finite(k::Kernel, x::AV) = finite(k, x)
lhs_finite(k::CrossKernel, x::AV) = LhsFiniteCrossKernel(k, x)
rhs_finite(k::CrossKernel, x′::AV) = RhsFiniteCrossKernel(k, x′)
finite(k::CrossKernel, x::AV, x′::AV) = FiniteCrossKernel(k, x, x′)
