export GP

"""
    GP{Tm<:MeanFunction, Tk<:Kernel}

A Gaussian Process (GP) with known mean `m` and kernel `k`. A book-keeping object `gpc` is
also required, but only matters when composing `GP`s together.

# Zero Mean

If only two arguments are provided, assume the mean to be zero everywhere:
```jldoctest
julia> f = GP(Matern32(), GPC());

julia> x = randn(5);

julia> mean(f(x)) == zeros(5)
true

julia> cov(f(x)) == Stheno.pw(Matern32(), x)
true
```

### Constant Mean

If a `Real` is provided as the first argument, assume the mean function is constant with
that value
```jldoctest
julia> f = GP(5.0, EQ(), GPC());

julia> x = randn(5);

julia> mean(f(x)) == 5.0 .* ones(5)
true

julia> cov(f(x)) == Stheno.pw(EQ(), x)
true
```

### Custom Mean

Provide an arbitrary function to compute the mean:

```jldoctest
julia> f = GP(x -> sin(x) + cos(x / 2), RQ(3.2), GPC());

julia> x = randn(5);

julia> mean(f(x)) == sin.(x) .+ cos.(x ./ 2)
true

julia> cov(f(x)) == Stheno.pw(RQ(3.2), x)
true
```
"""
struct GP{Tm<:MeanFunction, Tk<:Kernel} <: AbstractGP
    m::Tm
    k::Tk
    n::Int
    gpc::GPC
    function GP{Tm, Tk}(m::Tm, k::Tk, gpc::GPC) where {Tm, Tk}
        gp = new{Tm, Tk}(m, k, next_index(gpc), gpc)
        gpc.n += 1
        return gp
    end
end
GP(m::Tm, k::Tk, gpc::GPC) where {Tm<:MeanFunction, Tk<:Kernel} = GP{Tm, Tk}(m, k, gpc)

GP(f, k::Kernel, gpc::GPC) = GP(CustomMean(f), k, gpc)
GP(m::Real, k::Kernel, gpc::GPC) = GP(ConstMean(m), k, gpc)
GP(k::Kernel, gpc::GPC) = GP(ZeroMean(), k, gpc)
GP(k::Kernel, m, gpc::GPC) = GP(m, k, gpc)

mean_vector(f::GP, x::AV) = ew(f.m, x)

cov(f::GP, x::AV) = pw(f.k, x)
cov_diag(f::GP, x::AV) = ew(f.k, x)

cov(f::GP, x::AV, x′::AV) = pw(f.k, x, x′)
cov_diag(f::GP, x::AV, x′::AV) = ew(f.k, x, x′)

function cov(f::GP, f′::GP, x::AV, x′::AV)
    return f === f′ ? cov(f, x, x′) : zeros(length(x), length(x′))
end
function cov_diag(f::GP, f′::GP, x::AV, x′::AV)
    return f === f′ ? cov_diag(f, x, x′) : zeros(length(x))
end
