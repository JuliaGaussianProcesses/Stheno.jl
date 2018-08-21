"""

"""
∇(f::GP) = GP(∇, f)

μ_p′(::typeof(∇), f) = DerivativeMean(mean(f))
k_p′(::typeof(∇), f) = DerivativeKernel(kernel(f))
k_p′p(::typeof(∇), f, fp::GP) = DerivativeLhsCross(kernel(f, fp))
k_pp′(fp::GP, ::typeof(∇), f) = DerivativeRhsCross(kernel(fp, f))
