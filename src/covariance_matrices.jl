import Base.BLAS: trsv
import PDMats: AbstractPDMat, invquad, dim
import Base: cov, logdet, full, size, chol, ==
export cov, invquad, AbstractPDMat

const __ϵ = 1e-9

"""
    StridedPDMatrix

A strided positive definite matrix, represented in terms of it's Cholesky factorization `U`.
"""
struct StridedPDMatrix{T<:Real} <: AbstractPDMat{T}
    U::UpperTriangular{T}
end
dim(Σ::StridedPDMatrix) = size(Σ.U, 1)
full(Σ::StridedPDMatrix) = Σ.U'Σ.U
logdet(Σ::StridedPDMatrix) = 2 * sum(log, view(Σ.U, diagind(Σ.U)))
invquad(Σ::StridedPDMatrix, x::AbstractVector) = sum(abs2, trsv('U', 'T', 'N', Σ.U.data, x))
chol(Σ::StridedPDMatrix) = Σ.U
==(Σ1::StridedPDMatrix, Σ2::StridedPDMatrix) = Σ1.U == Σ2.U

"""
    cov(d::Union{GP, Vector})

Compute the covariance matrix implied by a FiniteGP.
"""
function cov(d::Vector)
    pos, N = 1, map(dims, d)
    K = Matrix{Float64}(sum(N), sum(N))
    for q in eachindex(d), c in 1:N[q], p in eachindex(d)
        pos = broadcast_over_col(K, pos, N[p], kernel(d[p], d[q]), c)
    end
    return StridedPDMatrix(chol(Symmetric(K) + __ϵ * I))
end
cov(d::GP) = cov([d])

"""
    cov(d::Union{GP, Vector}, d′::Union{GP, Vector})

Compute the cross-covariance between each GP in `d` and `d′`.
"""
function cov(d::Vector, d′::Vector)
    pos, Dx, Dx′ = 1, map(dims, d), map(dims, d′)
    K = Matrix{Float64}(sum(Dx), sum(Dx′))
    for q in eachindex(d′), c in 1:Dx′[q], p in eachindex(d)
        pos = broadcast_over_col(K, pos, Dx[p], kernel(d[p], d′[q]), c)
    end
    return K
end
cov(d::GP, d′::GP) = cov([d], [d′])
cov(d::GP, d′::Vector) = cov([d], d′)
cov(d::Vector, d′::GP) = cov(d, [d′])

# Function barrier to improve performance of `cov`.
function broadcast_over_col(K, pos, Np, k, c)
    for r in 1:Np
        @inbounds K[pos] = k(r, c)
        pos += 1
    end
    return pos
end
