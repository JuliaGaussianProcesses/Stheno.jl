using FDM, Zygote, Distances, Random, LinearAlgebra, FillArrays, ToeplitzMatrices

@testset "zygote_rules" begin

# # Check FillArrays work as expected.
# let
#     @test Zygote.gradient(x->sum(Fill(x, 10)), randn())[1] == 10
#     @test Zygote.gradient(x->sum(Fill(x, (10, 3, 4))), randn())[1] == 10 * 3 * 4

#     # Test unary broadcasting gradients.
#     x = randn()
#     out, back = Zygote.forward(x->exp.(x), Fill(x, 10))
#     @test out isa Fill
#     @test out == Fill(exp(x), 10)
#     @test back(Ones(10))[1] isa Fill
#     @test back(Ones(10))[1] == Ones(10) .* exp(x)
#     @test back(ones(10))[1] isa Vector
#     @test back(ones(10))[1] == ones(10) .* exp(x)
# end

# # Check squared-euclidean distance implementation (AbstractMatrix)
# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

#     # Check sqeuclidean.
#     x, y = randn(rng, D), randn(rng, D)
#     f_el_1 = x->sqeuclidean(x, y)
#     @test all(abs.(Zygote.gradient(f_el_1, x)[1] .- FDM.grad(fdm, f_el_1, x)) .< 1e-8) 

#     f_el_2 = y->sqeuclidean(x, y)
#     @test all(abs.(Zygote.gradient(f_el_2, y)[1] .- FDM.grad(fdm, f_el_2, y)) .< 1e-8)

#     # Check binary colwise
#     X, Y = randn(rng, D, P), randn(rng, D, P)
#     f_col_1 = X->sum(colwise(SqEuclidean(), X, Y))
#     @test all(abs.(Zygote.gradient(f_col_1, X)[1] .- FDM.grad(fdm, f_col_1, X)) .< 1e-8)

#     f_col_2 = Y->sum(colwise(SqEuclidean(), X, Y))
#     @test all(abs.(Zygote.gradient(f_col_2, Y)[1] .- FDM.grad(fdm, f_col_2, Y)) .< 1e-8)
# end

# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

#     # Generate differing-length vectors for pairwise.
#     X, Y = randn(rng, D, P), randn(rng, D, Q)

#     # Check first argument of binary pairwise.
#     f_pw_1 = X->sum(pairwise(SqEuclidean(), X, Y))
#     @test all(Zygote.gradient(f_pw_1, X)[1] .- FDM.grad(fdm, f_pw_1, X) .< 1e-8)

#     # Check second argument of binary pairwise.
#     f_pw_2 = Y->sum(pairwise(SqEuclidean(), X, Y))
#     @test all(Zygote.gradient(f_pw_2, Y)[1] .- FDM.grad(fdm, f_pw_2, Y) .< 1e-8)

#     # Check unary pairwise.
#     @test Zygote.gradient(X->sum(pairwise(SqEuclidean(), X)), X)[1] ≈
#         Zygote.gradient(X->sum(pairwise(SqEuclidean(), X, X)), X)[1]
# end

# # Check squared-euclidean distance implementation (AbstractVector)
# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q = MersenneTwister(123456), 10, 9
#     x, y = randn(rng, P), randn(rng, Q)

#     # Check first argument of binary pairwise.
#     f = x->sum(pairwise(SqEuclidean(), x, y))
#     @test all(Zygote.gradient(f, x)[1] .- FDM.grad(fdm, f, x) .< 1e-8)

#     # Check second argument of binary pairwise.
#     f = y->sum(pairwise(SqEuclidean(), x, y))
#     @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

#     # Check unary pairwise.
#     @test Zygote.gradient(x->sum(pairwise(SqEuclidean(), x)), x)[1] ≈
#         Zygote.gradient(x->sum(pairwise(SqEuclidean(), x, x)), x)[1]
# end

# # Check that \ and / work.
# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q = MersenneTwister(123456), 10, 9
#     X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)

#     # \

#     # Check first argument sensitivity in matrix case.
#     f = X->sum(X \ Y)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in matrix case.
#     f = Y->sum(X \ Y)
#     @test all(Zygote.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

#     # Check first argument sensitivity in vector case.
#     f = X->sum(X \ y)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in vector case.
#     f = y->sum(X \ y)
#     @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

#     # /

#     # Check first argumeant sensitivity in matrix case.
#     f = X->sum(Y' / X)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in matrix case.
#     f = Y->sum(Y' / X)
#     @test all(Zygote.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

#     # Check first argument sensitivity in vector case.
#     f = X->sum(y' / X)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in vector case.
#     f = y->sum(y' / X)
#     @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)
# end

# # Check that Symmetric works as expected.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 7
#     A = randn(rng, P, P)

#     f = A->sum(Symmetric(A))
#     @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
# end

# # Check that `diag` behaves sensibly.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 10
#     A = randn(rng, P, P)

#     f = A->sum(diag(A))
#     @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
# end

# # Verify that Cholesky retrieves gradient information correctly.
# let
#     rng, N = MersenneTwister(123456), 2
#     A, uplo, info = randn(rng, N, N), :U, 0

#     _, back = Zygote.forward(Cholesky, A, uplo, info)
#     @test back((factors=A, uplo=nothing, info=nothing)) == (A, nothing, nothing)
# end

# # Verify Cholesky factorisation correctness.
# let
#     rng, N = MersenneTwister(123456), 5
#     A = randn(rng, N, N)
#     S = A' * A + 1e-6I
#     fdm = central_fdm(5, 1)

#     # Check factorisation basics.
#     let
#         f = A->logdet(cholesky(Symmetric(A' * A + 1e-3I)))
#         @test all(abs.(Zygote.gradient(f, A)[1] - FDM.grad(fdm, f, A)) .< 1e-8)
#     end
#     let
#         f = A->sum(cholesky(Symmetric(A' * A + 1e-3I)).U)
#         @test all(Zygote.gradient(f, A)[1] - FDM.grad(fdm, f, A) .< 1e-8)
#     end
# end

# # Verify Cholesky factorisation `getproperty`.
# let
#     rng, N = MersenneTwister(123456), 5
#     A = randn(rng, N, N)
#     S = A' * A + 1e-6I
#     C = cholesky(S)
#     fdm = central_fdm(5, 1)

#     # Check that non-differentiable ops run forwards and have `nothing` gradients.
#     _, back = Zygote.forward(C->C.info, C)
#     @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)
#     _, back = Zygote.forward(C->C.uplo, C)
#     @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)

#     # Unit test retrieving the factors.
#     @test_broken Zygote.gradient(C->sum(C.factors), C)[1].factors == ones(size(C))

#     # Integration testing with backprop and comparison with FDM.
#     let
#         f = A->sum(cholesky(A' * A + 1e-6I).factors)
#         Ā_fd = FDM.grad(fdm, f, A)
#         @test_broken all(abs.(Zygote.gradient(f, A)[1] .- Ā_fd) .< 1e-8)
#     end

#     # Integration testing for retrieving upper triangle when uplo == 'U'.
#     let
#         f = A->sum(cholesky(A' * A + 1e-6I).U)
#         Ā_ad = Zygote.gradient(f, A)[1]
#         Ā_fd = FDM.grad(fdm, f, A)
#         @test all(abs.(Ā_ad .- Ā_fd) .< 1e-8)
#     end

#     # Integration testing for retrieving lower triangle when uplo == 'U'.
#     let
#         f = A->sum(cholesky(A' * A + 1e-6I).L)
#         Ā_ad = Zygote.gradient(f, A)[1]
#         Ā_fd = FDM.grad(fdm, f, A)
#         @test all(abs.(Ā_ad .- Ā_fd) .< 1e-8)
#     end

#     # Integration testing for retrieving upper triangle when uplo == 'L'.
#     let
#         f = A->sum(Cholesky(A, :L, 0).U)
#         Ā_ad = Zygote.gradient(f, A)[1]
#         Ā_fd = FDM.grad(fdm, f, A)
#         @test all(abs.(Ā_ad .- Ā_fd) .< 1e-8)
#     end

#     # Integration testing for retrieving lower triangle when uplo == 'L'.
#     let
#         f = A->sum(Cholesky(A, :L, 0).L)
#         Ā_ad = Zygote.gradient(f, A)[1]
#         Ā_fd = FDM.grad(fdm, f, A)
#         @test all(abs.(Ā_ad .- Ā_fd) .< 1e-8)
#     end
# end

# Verify Cholesky factorisation of Toeplitz matrices correctness.
let
    x = pairwise(EQ(), range(-3.0, stop=3.0, length=10))[:, 1]
    rng, N = MersenneTwister(123456), 4
    A = randn(rng, N, N)
    fdm = central_fdm(5, 1)

    # Check that forwards-passes agree.
    T = SymmetricToeplitz(x)
    @test Zygote.forward(cholesky, T)[1].L ≈ Stheno._cholesky(T).L

    # Slightly iffy unit testing for cholesky.
    let
        x = pairwise(EQ(), range(-3.0, stop=3.0, length=10))[:, 1]
        x[1] += 0.1
        f = x->sum(cholesky(SymmetricToeplitz(x)).U)

        x̄_ad = Zygote.gradient(f, x)[1]
        println("ad")
        display(x̄_ad)
        println()
        x̄_fd = FDM.grad(fdm, x->sum(Stheno._cholesky(SymmetricToeplitz(x)).U), x)
        println("fd")
        display(x̄_fd)
        println()
        @test all(abs.(x̄_ad .- x̄_fd) .< 1e-8)
    end

    # # An integration test for adjoint.
    # let
    #     f = function(A)
    #         S = A' * A + 1e-6I
    #         return sum(cholesky(SymmetricToeplitz(S[:, 1])).U)
    #     end
    #     x̄_ad = Zygote.gradient(f, A)[1]
    #     @show x̄_ad
    #     x̄_fd = FDM.grad(fdm, f, A)
    #     @show x̄_fd
    #     @test all(abs.(x̄_ad .- x̄_fd) .< 1e-8)
    # end
end

# # Check that addition of matrices and uniform scalings works as hoped.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 10
#     A = randn(rng, P, P)

#     f = A->sum(A + 5I)
#     @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
# end

end
