using ToeplitzMatrices
import LinearAlgebra: Symmetric

import Base: transpose, adjoint, copy
import ToeplitzMatrices: Toeplitz

function copy(T::Toeplitz)
    return Toeplitz(copy(T.vc), copy(T.vr), copy(T.vcvr_dft), copy(T.tmp), T.dft)
end
function copy(T::SymmetricToeplitz)
    return SymmetricToeplitz(copy(T.vc), copy(T.vcvr_dft), copy(T.tmp), T.dft)
end


"""
    chol!(L::AbstractMatrix, T::SymmetricToeplitz)

Implementation adapted from "On the stability of the Bareiss and related
Toeplitz factorization algorithms", Bojanczyk et al, 1993.
"""
function chol!(L::AbstractMatrix, T::SymmetricToeplitz)

    # Initialize.
    L[:, 1] .= T.vc ./ sqrt(T.vc[1])
    v = copy(L[:, 1])
    N = size(T, 1)

    # Iterate.
    @inbounds for n in 1:N-1
        sinθn = v[n + 1] / L[n, n]
        cosθn = sqrt(1 - sinθn^2)

        for n′ in n+1:N
            v[n′] = (v[n′] - sinθn * L[n′ - 1, n]) / cosθn
            L[n′, n + 1] = -sinθn * v[n′] + cosθn * L[n′ - 1, n]
        end
    end
    return UpperTriangular(L')
end
chol(T::SymmetricToeplitz) = chol!(Matrix{eltype(T)}(size(T, 1), size(T, 1)), T)

function +(T::SymmetricToeplitz, u::UniformScaling)
    Tu = copy(T)
    Tu.vc[1] += u.λ
    Tu.vcvr_dft .+= u.λ
    return Tu
end
+(u::UniformScaling, T::SymmetricToeplitz) = T + u

transpose(T::Toeplitz) = Toeplitz(T.vr, T.vc)
adjoint(T::Toeplitz) = Toeplitz(conj.(T.vr), conj.(T.vc))
transpose(T::SymmetricToeplitz) = T
adjoint(T::SymmetricToeplitz) = T

@inline LinearAlgebra.Symmetric(T::SymmetricToeplitz) = T

Toeplitz(vc::AbstractVector, vr::AbstractVector) = Toeplitz(Vector(vc), Vector(vr))
