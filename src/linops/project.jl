export project

"""
    project(ϕ, f(x)::FiniteGP, g)

The projection operation `f′(x) = Σₘ ϕₘ(x) f(m) + g(x)`.
"""
project(ϕ, f::FiniteGP, g::MeanFunction) = GP(project, ϕ, f.f, g, f.x)
project(ϕ, f::GP) = project(ϕ, f, ZeroMean())

function μ_p′(::typeof(project), ϕ, f, g, x)
    return iszero(mean(f)) ? g : DeltaSumMean(ϕ, mean(f), g, x)
end
function k_p′(::typeof(project), ϕ, f, g, x)
    return iszero(kernel(f)) ? ZeroKernel() : DeltaSumKernel(ϕ, kernel(f), x)
end
function k_p′p(::typeof(project), ϕ, f, g, x, fp)
    return iszero(kernel(f, fp)) ? ZeroKernel() : LhsDeltaSumCrossKernel(ϕ, kernel(f, fp), x)
end
function k_pp′(fp, ::typeof(project), ϕ, f, g, x)
    return iszero(kernel(f, fp)) ? ZeroKernel() : RhsDeltaSumCrossKernel(kernel(f, fp), ϕ, x)
end
