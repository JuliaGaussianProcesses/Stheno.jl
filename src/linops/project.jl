export project

"""
    project(ϕ, f::GP, Z::AVM, g)

The projection operation `f′(x) = Σ_j ϕ(x, Z[j]) * f(Z[j]) + g(x)`.
"""
project(ϕ, f::GP, Z::AV, g) = GP(project, ϕ, f, Z, g)

μ_p′(::typeof(project), ϕ, f, Z, g) = DeltaSumMean(ϕ, FiniteMean(mean(f), Z), g)
k_p′(::typeof(project), ϕ, f, Z, g) = DeltaSumKernel(ϕ, FiniteKernel(kernel(f), Z))
function k_p′p(::typeof(project), ϕ, f, Z, g, fp)
    return LhsDeltaSumCrossKernel(ϕ, LhsFiniteCrossKernel(kernel(f, fp), Z))
end
function k_pp′(fp, ::typeof(project), ϕ, f, Z, g)
    return RhsDeltaSumCrossKernel(LhsFiniteCrossKernel(kernel(f, fp), Z), ϕ)
end

"""
    project(ϕ, f::GP, g)

The projection operation `f′(x) = Σₘ ϕₘ(x) f(m) + g(x)`. Assumes `f` is finite.
"""
function project(ϕ, f::GP, g)
    @assert isfinite(length(f))
    return GP(project, ϕ, f, :, g)
end

project(ϕ, f::GP) = project(ϕ, f, ZeroMean{Float64}())
