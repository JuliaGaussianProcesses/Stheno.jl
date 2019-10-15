using Kronecker

"""
    RectilinearGrid{T, Txl, Txr} <: AbstractVector{T}

A vector of length `length(xl) * length(xr)` which represents a matrix of data of size
`(length(xl), length(xr))`, the `ij`th element of which is the point `(xl[i], xr[j])`.
"""
struct RectilinearGrid{T, Txl, Txr} <: AbstractVector{T}
    xl::Txl
    xr::Txr
    function RectilinearGrid(xl::AV{T}, xr::AV{V}) where {T, V}
        return new{promote_type(T, V), typeof(xl), typeof(xr)}(xl, xr)
    end
end
Base.size(x::RectilinearGrid) = (length(x),)
Base.length(x::RectilinearGrid) = length(x.xl) * length(x.xr)


"""
    Separable{Tkl, Tkr} <: Kernel

A kernel that is separable over two input dimensions
"""
struct Separable{Tkl, Tkr} <: Kernel
    kl::Tkl
    kr::Tkr
end

pw(k::Separable, x::RectilinearGrid) = pw(k.kl, x.xl) ⊗ pw(k.kr, x.xr)
ew(k::Separable, x::RectilinearGrid) = error("Not implemented")

pw(k::Separable, x::RectilinearGrid, x′::RectilinearGrid) = error("Not implemented")
ew(k::Separable, x::RectilinearGrid, x′::RectilinearGrid) = error("Not implemented")

const SeparableGP = GP{<:MeanFunction, <:Separable}
const SeparableFiniteGP = FiniteGP{<:SeparableGP, <:RectilinearGrid, <:Diagonal}

function logpdf(f::SeparableFiniteGP, y::AV{<:Real})

    # Check that data and grid are the same lengths.
    @assert length(f.x) == length(y)

    # Check that the observation noise is isotropic. Ideally move this to compile-time
    # at some point,, although likely not a bottleneck.
    σ²_n = first(f.Σy.diag)
    @assert all(f.Σy.diag .== σ²_n)

    # Compute log marginal likelihood. A little bit uglier than I would like. Ideally we
    # would just write `eigen(cov(f))` and the correct thing would happen, but the
    # things aren't currently implemented correctly for this to be the case.
    K_eig = eigen(cov(f.f, f.x)) + σ²_n * I

    λ, Γ = K_eig
    β = Diagonal(1 ./ sqrt.(λ)) * (Γ'y)
    return -(length(y) * log(2π) + logdet(K_eig) + sum(abs2, β)) / 2
end

function rand(rng::AbstractRNG, f::SeparableFiniteGP, N::Int)

    # Check that the observation noise is isotropic. Ideally move this to compile-time
    # at some point,, although likely not a bottleneck.
    σ²_n = first(f.Σy.diag)
    @assert all(f.Σy.diag .== σ²_n)

    # Compute log marginal likelihood. A little bit uglier than I would like. Ideally we
    # would just write `eigen(cov(f))` and the correct thing would happen, but the
    # things aren't currently implemented correctly for this to be the case.
    K_eig = eigen(cov(f.f, f.x)) + σ²_n * I

    λ, Γ = K_eig
    return Γ * (Diagonal(sqrt.(λ)) * randn(rng, length(λ), N))
end
