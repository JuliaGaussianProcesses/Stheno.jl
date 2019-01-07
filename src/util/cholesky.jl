# Implementation due to Seeger, Matthias, et al. "Auto-differentiating linear algebra."
@adjoint function cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}})
    C = cholesky(Σ)
    U = C.U
    return C, function(Δ)
        Ū = Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
        @inbounds for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄),)
    end
end

# Various sensitivities for `literal_getproperty`, depending on the 2nd argument.
@adjoint function literal_getproperty(C::Cholesky, ::Val{:uplo})
    return literal_getproperty(C, Val(:uplo)), function(Δ)
        return ((uplo=nothing, info=nothing, factors=nothing),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:info})
    return literal_getproperty(C, Val(:info)), function(Δ)
        return ((uplo=nothing, info=nothing, factors=nothing),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:U})
    return literal_getproperty(C, Val(:U)), function(Δ)
        Δ_factors = C.uplo == 'U' ? UpperTriangular(Δ) : LowerTriangular(copy(Δ'))
        return ((uplo=nothing, info=nothing, factors=Δ_factors),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:L})
    return literal_getproperty(C, Val(:L)), function(Δ)
        Δ_factors = C.uplo == 'L' ? LowerTriangular(Δ) : UpperTriangular(copy(Δ'))
        return ((uplo=nothing, info=nothing, factors=Δ_factors),)
    end
end

@adjoint function logdet(C::Cholesky)
    return logdet(C), function(Δ)
        return ((uplo=nothing, info=nothing, factors=Diagonal(2 .* Δ ./ diag(C.factors))),)
    end
end
