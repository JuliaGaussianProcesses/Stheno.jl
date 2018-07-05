export project

"""
    project(ϕ, f::GP, g)

The projection operation `f′(x) = Σₘ ϕₘ(x) f(m) + g(x)`. Assumes `f` is finite.
"""
function project(ϕ, f::GP, g)
    @assert isfinite(length(f))
    return GP(project, ϕ, f, g)
end

project(ϕ, f::GP) = project(ϕ, f, ZeroMean{Float64}())

μ_p′(::typeof(project), ϕ, f, g) = DeltaSumMean(ϕ, mean(f), g)
k_p′(::typeof(project), ϕ, f, g) = DeltaSumKernel(ϕ, kernel(f))
k_p′p(::typeof(project), ϕ, f, g, fp) = LhsDeltaSumCrossKernel(ϕ, kernel(f, fp))
k_pp′(fp, ::typeof(project), ϕ, f, g) = RhsDeltaSumCrossKernel(kernel(f, fp), ϕ)
