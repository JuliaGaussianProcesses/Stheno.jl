export GPC

# A collection of GPs (GPC == "GP Collection"). Used to keep track of GPs.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

@nograd GPC

next_index(gpc::GPC) = gpc.n + 1

#
# Projecting AbstractGPs onto a finite dimensional marginal
#

import Base: rand, length
import Distributions: logpdf, AbstractMvNormal

export mean, std, cov, marginals, rand, logpdf, elbo, dtc

"""
    cov(fx::FiniteGP, gx::FiniteGP)

Compute the cross-covariance matrix between `fx` and `gx`.

```jldoctest
julia> f = wrap(GP(Matern32Kernel()), GPC());

julia> x1 = randn(11);

julia> x2 = randn(13);

julia> cov(f(x1), f(x2)) == kernelmatrix(Matern32Kernel(), x1, x2)
true
```
"""
cov(fx::FiniteGP, gx::FiniteGP) = cov(fx.f, gx.f, fx.x, gx.x)

# """
#     rand(rng::AbstractRNG, f::FiniteGP, N::Int=1)

# Obtain `N` independent samples from the marginals `f` using `rng`. Single-sample methods
# produce a `length(f)` vector. Multi-sample methods produce a `length(f)` x `N` `Matrix`.

# ```jldoctest
# julia> f = wrap(GP(Matern32Kernel()), GPC());

# julia> x = randn(11);

# julia> rand(f(x)) isa Vector{Float64}
# true

# julia> rand(MersenneTwister(123456), f(x)) isa Vector{Float64}
# true

# julia> rand(f(x), 3) isa Matrix{Float64}
# true

# julia> rand(MersenneTwister(123456), f(x), 3) isa Matrix{Float64}
# true
# ```
# """
# function rand(rng::AbstractRNG, f::FiniteGP, N::Int)
#     μ, C = mean(f), cholesky(Symmetric(cov(f)))
#     return μ .+ C.U' * randn(rng, promote_type(eltype(μ), eltype(C)), length(μ), N)
# end
# rand(f::FiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
# rand(rng::AbstractRNG, f::FiniteGP) = vec(rand(rng, f, 1))
# rand(f::FiniteGP) = vec(rand(f, 1))

# TYPE PIRACY!
LinearAlgebra.cholesky(D::Diagonal{<:Real, <:Fill}) = AbstractGPs._cholesky(D)


# function consistency_check(f, y, u)
#     @assert length(f) == size(y, 1)
# end
# Zygote.@nograd consistency_check

# import Base: \ 
# \(A::AbstractMatrix, B::Diagonal) = A \ Matrix(B)

# \(A::Union{LowerTriangular, UpperTriangular}, B::Diagonal) = A \ Matrix(B)
# \(A::Adjoint{<:Any, <:Union{LowerTriangular, UpperTriangular}}, B::Diagonal) = A \ Matrix(B)


# # Compute tr(Cf / Σy) efficiently for different types of Σy. For dense Σy you obviously need
# # to compute the entirety of Cf, which is bad, but for particular structured Σy one requires
# # only a subset of the elements. Σy isa UniformScaling is version usually considered.
# function tr_Cf_invΣy(f::FiniteGP, Σy::UniformScaling, chol_Σy::Cholesky)
#     return sum(cov_diag(f.f, f.x)) / Σy.λ
# end
# function tr_Cf_invΣy(f::FiniteGP, Σy::Diagonal, chol_Σy::Cholesky)
#     return sum(cov_diag(f.f, f.x) ./ diag(Σy))
# end
# tr_Cf_invΣy(f::FiniteGP, Σy::Matrix, chol_Σy::Cholesky) = tr(chol_Σy \ cov(f))

_test_block_consistency(ids, f) = length.(ids) == length.(blocks(f.x))
Zygote.@nograd _test_block_consistency

import Base: |, merge
export ←, |, Obs

"""
    Observation

Represents fixing a paricular (finite) GP to have a particular (vector) value.
"""
struct Observation{Tf<:FiniteGP, Ty<:Vector}
    f::Tf
    y::Ty
end

const Obs = Observation


←(f, y) = Observation(f, y)
get_f(c::Observation) = c.f
get_y(c::Observation) = c.y
