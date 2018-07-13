using ToeplitzMatrices

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
    return LowerTriangular(L)
end
chol(T::SymmetricToeplitz) = chol!(Matrix{eltype(T)}(size(T, 1), size(T, 1)), T)

function +(T::SymmetricToeplitz, u::UniformScaling)
    t = copy(T.vc)
    t[1] += u.λ
    return SymmetricToeplitz(t)
end
