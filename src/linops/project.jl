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

function μ_p′(::typeof(project), ϕ, f, g)
    return iszero(mean(f)) ? g : DeltaSumMean(ϕ, mean(f), g)
end
function k_p′(::typeof(project), ϕ, f, g)
    return iszero(kernel(f)) ? g : DeltaSumKernel(ϕ, kernel(f))
end
function k_p′p(::typeof(project), ϕ, f, g, fp)
    return iszero(kernel(f, fp)) ? g : LhsDeltaSumCrossKernel(ϕ, kernel(f, fp))
end
function k_pp′(fp, ::typeof(project), ϕ, f, g)
    return iszero(kernel(f, fp)) ? g : RhsDeltaSumCrossKernel(kernel(f, fp), ϕ)
end
