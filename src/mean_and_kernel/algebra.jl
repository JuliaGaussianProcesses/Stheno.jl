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

# Adding things to (Cross)Kernels.
+(α::Real, k::CrossKernel) = BinaryCrossKernel(+, α * OneKernel(), k)
+(k::CrossKernel, α::Real) = BinaryCrossKernel(+, k, OneKernel() * α)
+(α::Real, k::Kernel) = BinaryKernel(+, α * OneKernel(), k)
+(k::Kernel, α::Real) = BinaryKernel(+, k, OneKernel() * α)
+(k::CrossKernel, k′::CrossKernel) = BinaryCrossKernel(+, k, k′)
+(k::Kernel, k′::Kernel) = BinaryKernel(+, k, k′)

# Optimise addition with zero.
+(k::ZeroKernel, k′::ZeroKernel) = k
+(k::ZeroKernel, k′::CrossKernel) = k′
+(k::CrossKernel, k′::ZeroKernel) = k
+(k::ZeroKernel, k′::Kernel) = k′
+(k::Kernel, k′::ZeroKernel) = k

# Create finite-dimensional kernels.
finite(k::Kernel, x::AV) = finite(k, x)
lhs_finite(k::CrossKernel, x::AV) = LhsFiniteCrossKernel(k, x)
rhs_finite(k::CrossKernel, x′::AV) = RhsFiniteCrossKernel(k, x′)
finite(k::CrossKernel, x::AV, x′::AV) = FiniteCrossKernel(k, x, x′)


# ############################## Convenience functionality ##############################

# import Base: +, *, promote_rule, convert

# function *(f, μ::MeanFunction)
#     return iszero(μ) ? μ : CompositeMean(f, μ)
# end

# function *(f, k::CrossKernel)
#     return iszero(k) ? k : LhsCross(f, k)
# end
# function *(k::CrossKernel, f)
#     return iszero(k) ? k : RhsCross(k, f)
# end
# function *(f, k::RhsCross)
#     if k.k isa Kernel && f == k.f
#         return OuterKernel(f, k.k)
#     else
#         return LhsCross(f, k)
#     end
# end
# function *(k::LhsCross, f)
#     if k.k isa Kernel && f == k.f
#         return OuterKernel(f, k.k)
#     else
#         return RhsCross(k, f)
#     end
# end

# promote_rule(::Type{<:MeanFunction}, ::Type{<:Real}) = MeanFunction
# convert(::Type{MeanFunction}, x::Real) = ConstantMean(x)

# promote_rule(::Type{<:Kernel}, ::Type{<:Real}) = Kernel
# convert(::Type{<:CrossKernel}, x::Real) = ConstantKernel(x)

# # Composing mean functions with Reals.
# +(μ::MeanFunction, μ′::Real) = +(promote(μ, μ′)...)
# +(μ::Real, μ′::MeanFunction) = +(promote(μ, μ′)...)

# *(μ::MeanFunction, μ′::Real) = *(promote(μ, μ′)...)
# *(μ::Real, μ′::MeanFunction) = *(promote(μ, μ′)...)

# # Composing kernels with Reals.
# +(k::CrossKernel, k′::Real) = +(promote(k, k′)...)
# +(k::Real, k′::CrossKernel) = +(promote(k, k′)...)

# *(k::CrossKernel, k′::Real) = *(promote(k, k′)...)
# *(k::Real, k′::CrossKernel) = *(promote(k, k′)...)


# import Base: zero, +, *

# const ZM = ZeroMean{Float64}
# zero(::Type{<:MeanFunction}, N::Int) = FiniteZeroMean(1:N)
# zero(::Type{<:MeanFunction}) = ZeroMean{Float64}()
# zero(μ::MeanFunction) = length(μ) < Inf ? FiniteZeroMean(eachindex(μ)) : ZM()

# # Addition of mean functions.
# function +(μ::MeanFunction, μ′::MeanFunction)
#     if iszero(μ)
#         return μ′
#     elseif iszero(μ′)
#         return μ
#     else
#         return BinaryMean(+, μ, μ′)
#     end
# end
# +(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c + μ′.c)
# +(a::Real, μ::MeanFunction) = UnaryMean(m->a + m, μ)
# +(μ::MeanFunction, a::Real) = UnaryMean(m->m + a, μ)

# # Product of mean functions.
# function *(μ::MeanFunction, μ′::MeanFunction)
#     if iszero(μ)
#         return μ
#     elseif iszero(μ′)
#         return μ′
#     else
#         return BinaryMean(*, μ, μ′)
#     end
# end
# *(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c * μ′.c)
# # *(a::Real, μ::MeanFunction) = UnaryMean(m->a * m, μ)
# # *(μ::MeanFunction, a::Real) = UnaryMean(m->m * a, μ)

# @inline *(a::Real, μ::MeanFunction) = BinaryMean(*, a, μ)
# @inline *(μ::MeanFunction, a::Real) = BinaryMean(*, μ, a)

# import Base: iszero, zero, *

# # Define addition of zeros.
# for T in [
#     :FiniteZeroKernel,
#     :LhsFiniteZeroCrossKernel,
#     :RhsFiniteZeroCrossKernel,
#     :FiniteZeroCrossKernel,
# ]
#     @eval +(x::$T, x′::$T) = zero(x)
# end

# # Define zeros for block stuff.
# zero(μ::BlockMean) = BlockMean(zero.(μ.μ))
# zero(k::BlockCrossKernel) = BlockCrossKernel(zero.(k.ks))
# zero(k::BlockKernel) = BlockKernel(zero.(k.ks_diag), zero.(k.ks_off))
