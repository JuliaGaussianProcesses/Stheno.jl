export ∫, StandardNormal

struct StandardNormal end

"""
    ∫(::StandardNormal, f::GP{<:SEKernel})

Return the scalar-valued process which results from computing the expectation of `f`, whose
mean and kernel are the zero-function and `SEKernel` resp., under a standard normal distribution.
"""
∫(::StandardNormal, f::GP{<:μFun, <:SEKernel}) = GP(∫, StandardNormal(), f)
μ_p′(::typeof(∫), ::StandardNormal, f::GP{<:μFun, <:SEKernel}) = ZeroMean()
k_p′(::typeof(∫), ::StandardNormal, f::GP{<:μFun, <:SEKernel}) = Finite(Constant(1 / sqrt(3)), [1])
k_pp′(f_p::GP, ::typeof(∫), ::StandardNormal, f::GP) =
    f_p === f ?
        RhsFinite((x, ::Any)->exp(-0.25 * x^2) / sqrt(2), [1]) :
        throw(error("Don't know how to compute xcovs for general graphs for integration."))
k_p′p(f_p::GP, ::typeof(∫), ::StandardNormal, f::GP) =
    f_p === f ?
        LhsFinite((::Any, x′)->exp(-0.25 * x′^2) / sqrt(2), [1]) :
        throw(error("Don't know how to compute xcovs for general graphs for integration."))
