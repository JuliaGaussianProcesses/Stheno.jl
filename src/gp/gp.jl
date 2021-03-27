export GP

"""
    GP{Tm<:MeanFunction, Tk<:Kernel}

A Gaussian Process (GP) with known mean `m` and kernel `k`. A book-keeping object `gpc` is
also required, but only matters when composing `GP`s together.

# Zero Mean

If only two arguments are provided, assume the mean to be zero everywhere:
```jldoctest
julia> f = GP(Matern32Kernel(), GPC());

julia> x = randn(5);

julia> mean(f(x)) == zeros(5)
true

julia> cov(f(x)) == kernelmatrix(Matern32Kernel(), x)
true
```

### Constant Mean

If a `Real` is provided as the first argument, assume the mean function is constant with
that value
```jldoctest
julia> f = GP(5.0, SqExponentialKernel(), GPC());

julia> x = randn(5);

julia> mean(f(x)) == 5.0 .* ones(5)
true

julia> cov(f(x)) == kernelmatrix(SqExponentialKernel(), x)
true
```

### Custom Mean

Provide an arbitrary function to compute the mean:

```jldoctest
julia> f = GP(x -> sin(x) + cos(x / 2), RationalQuadraticKernel(α=3.2), GPC());

julia> x = randn(5);

julia> mean(f(x)) == sin.(x) .+ cos.(x ./ 2)
true

julia> cov(f(x)) == kernelmatrix(RationalQuadraticKernel(α=3.2), x)
true
```
"""
struct GP{Tm<:MeanFunction, Tk<:Kernel} <: AbstractGP
    m::Tm
    k::Tk
end

GP(f, k::Kernel) = GP(CustomMean(f), k)
GP(m::Real, k::Kernel) = GP(ConstMean(m), k)
GP(k::Kernel) = GP(ZeroMean(), k)
GP(k::Kernel, m) = GP(m, k)

mean_vector(f::GP, x::AV) = elementwise(f.m, x)

cov(f::GP, x::AV) = kernelmatrix(f.k, x)
cov_diag(f::GP, x::AV) = kernelmatrix_diag(f.k, x)

cov(f::GP, x::AV, x′::AV) = kernelmatrix(f.k, x, x′)
cov_diag(f::GP, x::AV, x′::AV) = kernelmatrix_diag(f.k, x, x′)

"""
    WrappedGP{Tgp} <: AbstractGP

A thin wrapper around a GP that does some book-keeping.
"""
struct WrappedGP{Tgp} <: AbstractGP
    gp::Tgp
    n::Int
    gpc::GPC
    function WrappedGP{Tgp}(gp::Tgp, gpc::GPC) where {Tgp<:GP}
        wgp = new{Tgp}(gp, next_index(gpc), gpc)
        gpc.n += 1
        return wgp
    end
end

wrap(gp::Tgp, gpc::GPC) where {Tgp<:GP} = WrappedGP{Tgp}(gp, gpc)

mean_vector(f::WrappedGP, x::AV) = mean_vector(f.gp, x)

cov(f::WrappedGP, x::AV) = cov(f.gp, x)
cov_diag(f::WrappedGP, x::AV) = cov_diag(f.gp, x)

cov(f::WrappedGP, x::AV, x′::AV) = cov(f.gp, x, x′)
cov_diag(f::WrappedGP, x::AV, x′::AV) = cov_diag(f.gp, x, x′)

function cov(f::WrappedGP, f′::WrappedGP, x::AV, x′::AV)
    return f === f′ ? cov(f, x, x′) : zeros(length(x), length(x′))
end
function cov_diag(f::WrappedGP, f′::WrappedGP, x::AV, x′::AV)
    return f === f′ ? cov_diag(f, x, x′) : zeros(length(x))
end
