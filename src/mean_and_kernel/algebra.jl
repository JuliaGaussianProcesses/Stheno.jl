import Base: +, *
export finite


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

##################
# Kernel algebra #
##################

# Adding things to Kernels.
+(α::Real, k::Kernel) = BinaryKernel(+, ConstKernel(α), k)
+(k::Kernel, α::Real) = BinaryKernel(+, k, ConstKernel(α))
+(k::Kernel, k′::Kernel) = BinaryKernel(+, k, k′)

# Optimise addition with zero.
+(k::ZeroKernel, k′::ZeroKernel) = k
+(k::ZeroKernel, k′::Kernel) = k′
+(k::Kernel, k′::ZeroKernel) = k

# Optimise for addition of `ConstKernel`s.
+(k::ConstKernel, k′::ConstKernel) = ConstKernel(k.c + k′.c)

# Multiply kernels by things.
*(k::Kernel, k′::Kernel) = BinaryKernel(*, k, k′)
*(m::MeanFunction, k::Kernel) = LhsCross(m, k)
*(k::Kernel, m::MeanFunction) = RhsCross(k, m)
*(α::Real, k::Kernel) = ConstMean(α) * k
*(k::Kernel, α::Real) = k * ConstMean(α)
*(f, k::Kernel) = CustomMean(f) * k
*(k::Kernel, f) = k * CustomMean(f)

# Optimise multiplication by zeros.
*(k::ZeroKernel, k′::ZeroKernel) = k
*(k::ZeroKernel, k′::Kernel) = k
*(k::Kernel, k′::ZeroKernel) = k′
*(k::ZeroKernel, m::MeanFunction) = k
*(m::MeanFunction, k::ZeroKernel) = k

# Optimise multiplication by one.
*(k::OneKernel, k′::OneKernel) = k
*(k::OneKernel, k′::Kernel) = k′
*(k::Kernel, k′::OneKernel) = k

# Optimise for product of `ConstKernel`s.
*(k::ConstKernel, k′::ConstKernel) = ConstKernel(k.c * k′.c)

#######################
# CrossKernel algebra #
#######################

# Adding things to (Cross)Kernels.
+(α::Real, k::CrossKernel) = BinaryCrossKernel(+, ConstKernel(α), k)
+(k::CrossKernel, α::Real) = BinaryCrossKernel(+, k, ConstKernel(α))
+(k::CrossKernel, k′::CrossKernel) = BinaryCrossKernel(+, k, k′)

# Optimise addition with zero.
+(k::ZeroKernel, k′::CrossKernel) = k′
+(k::CrossKernel, k′::ZeroKernel) = k

# Multiply kernels by things.
*(k::CrossKernel, k′::CrossKernel) = BinaryCrossKernel(*, k, k′)
*(m::MeanFunction, k::CrossKernel) = LhsCross(m, k)
*(k::CrossKernel, m::MeanFunction) = RhsCross(k, m)
*(α::Real, k::CrossKernel) = ConstMean(α) * k
*(k::CrossKernel, α::Real) = k * ConstMean(α)
*(f, k::CrossKernel) = CustomMean(f) * k
*(k::CrossKernel, f) = k * CustomMean(f)

# Optimise multiplication by zeros.
*(k::ZeroKernel, k′::CrossKernel) = k
*(k::CrossKernel, k′::ZeroKernel) = k′

# Optimise multiplication by one.
*(k::OneKernel, k′::CrossKernel) = k′
*(k::CrossKernel, k′::OneKernel) = k
