# The thing that most internal types should probably subtype.
abstract type SthenoAbstractGP <: AbstractGP end

# TYPE-PIRACY
function AbstractGPs.cov_diag(f::GP, x::AbstractVector, x′::AbstractVector)
    return kernelmatrix_diag(f.kernel, x, x′)
end

# Implement some of the AbstractGPs API for all of the GPs in this package.
mean_and_cov(f::SthenoAbstractGP, x::AbstractVector) = (mean(f, x), cov(f, x))

mean_and_cov_diag(f::SthenoAbstractGP, x::AbstractVector) = (mean(f, x), cov_diag(f, x))

# Ensure that this package gets to handle the covariance between its own GPs.
# AbstractGPs doesn't support this in general because it's unclear how it ought to be
# implemented, but we have a clear way to implement it here.
function cov(fx::FiniteGP{<:SthenoAbstractGP}, gx::FiniteGP{<:SthenoAbstractGP})
    return cov(fx.f, gx.f, fx.x, gx.x)
end


# A collection of GPs (GPC == "GP Collection"). Used to keep track of GPs.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

ChainRulesCore.@non_differentiable GPC()

next_index(gpc::GPC) = gpc.n + 1

# TYPE PIRACY!
LinearAlgebra.cholesky(D::Diagonal{<:Real, <:Fill}) = AbstractGPs._cholesky(D)
