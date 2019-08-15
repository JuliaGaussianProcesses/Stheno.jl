using FiniteDifferences, Zygote, Distances, Random, LinearAlgebra, StatsFuns
using Base.Broadcast: broadcast_shape

@testset "zygote_rules" begin
    @testset "Cholesky (ctor)" begin
        rng, N = MersenneTwister(123456), 2
        A, uplo, info = randn(rng, N, N), :U, 0
        _, back = Zygote.forward(Cholesky, A, uplo, info)
        @test back((factors=A, uplo=nothing, info=nothing)) == (A, nothing, nothing)
    end
    @testset "Cholesky (getproperty)" begin
        rng, N = MersenneTwister(123456), 5
        A = randn(rng, N, N)
        S = A' * A + 1e-6I
        C = cholesky(S)

        # Check that non-differentiable ops run forwards and have `nothing` gradients.
        _, back = Zygote.forward(C->C.info, C)
        @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)
        _, back = Zygote.forward(C->C.uplo, C)
        @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)

        # Unit test retrieving the factors.
        @test_throws ErrorException Zygote.forward(C->C.factors, C)

        # Test getproperty.
        adjoint_test(A->Cholesky(A, :U, 0).U, randn(rng, N, N), A)
        adjoint_test(A->Cholesky(A, :U, 0).L, randn(rng, N, N), A)
        adjoint_test(A->Cholesky(A, :L, 0).U, randn(rng, N, N), A)
        adjoint_test(A->Cholesky(A, :L, 0).L, randn(rng, N, N), A)
    end
    @testset "cholesky (Matrix)" begin
        rng, N = MersenneTwister(123456), 3
        A = randn(rng, N, N)
        adjoint_test(A->logdet(cholesky(Symmetric(A' * A + 1e-3I))), randn(rng), A)
        adjoint_test(A->cholesky(Symmetric(A' * A + 1e-3I)).U, randn(rng, N, N), A)
        adjoint_test(A->cholesky(Symmetric(A' * A + 1e-3I)).L, randn(rng, N, N), A)

        adjoint_test(A->logdet(cholesky(A' * A + 1e-3I)), randn(rng), A)
        adjoint_test(A->cholesky(A' * A + 1e-3I).U, randn(rng, N, N), A)
        adjoint_test(A->cholesky(A' * A + 1e-3I).L, randn(rng, N, N), A)
    end
    @testset "colwise(::Euclidean, X, Y; dims=2)" begin
        rng, D, P = MersenneTwister(123456), 2, 3
        X, Y, D̄ = randn(rng, D, P), randn(rng, D, P), randn(rng, P)
        adjoint_test((X, Y)->colwise(Euclidean(), X, Y), D̄, X, Y)
    end
    @testset "pairwise(::Euclidean, X, Y; dims=2)" begin
        rng, D, P, Q = MersenneTwister(123456), 2, 3, 5
        X, Y, D̄ = randn(rng, D, P), randn(rng, D, Q), randn(rng, P, Q)
        adjoint_test((X, Y)->pairwise(Euclidean(), X, Y; dims=2), D̄, X, Y)
    end
    @testset "pairwise(::Euclidean, X; dims=2)" begin
        rng, D, P = MersenneTwister(123456), 2, 3
        X, D̄ = randn(rng, D, P), randn(rng, P, P)
        adjoint_test(X->pairwise(Euclidean(), X; dims=2), D̄, X)
    end
    @testset "Diagonal" begin
        rng, N = MersenneTwister(123456), 11
        adjoint_test(Diagonal, rand(rng, N, N), randn(rng, N))
        adjoint_test(x->Diagonal(x).diag, randn(rng, N), randn(rng, N))
    end
    @testset "fill" begin
        rng, N, M, P = MersenneTwister(123456), 11, 6, 5
        adjoint_test(x->fill(x, N), randn(rng, N), randn(rng))
        adjoint_test(x->fill(x, N, M), randn(rng, N, M), randn(rng))
        adjoint_test(x->fill(x, N, M, P), randn(rng, N, M, P), randn(rng))
    end
    @testset "xlogx" begin
        rng = MersenneTwister(123456)
        # adjoint_test(xlogx, randn(rng), -5.0)
        # adjoint_test(xlogx, randn(rng), 0.0; fdm=backward_fdm(5, 1))
        adjoint_test(xlogx, randn(rng), 1.0; fdm=forward_fdm(5, 1))
        adjoint_test(xlogx, randn(rng), 2.45)
    end
    @testset "logistic" begin
        rng = MersenneTwister(123456)
        adjoint_test(logistic, randn(rng), -5.0)
        adjoint_test(logistic, randn(rng), -1.0)
        adjoint_test(logistic, randn(rng), -eps())
        adjoint_test(logistic, randn(rng), 0.0)
        adjoint_test(logistic, randn(rng), eps())
        adjoint_test(logistic, randn(rng), 1.0)
        adjoint_test(logistic, randn(rng), 5.0)
    end
    @testset "logit" begin
        rng = MersenneTwister(123456)
        adjoint_test(logit, randn(rng), 0.1; fdm=forward_fdm(5, 1), atol=1e-7, rtol=1e-7)
        adjoint_test(logit, randn(rng), 0.3)
        adjoint_test(logit, randn(rng), 0.5)
        adjoint_test(logit, randn(rng), 0.7)
        adjoint_test(logit, randn(rng), 0.9; fdm=backward_fdm(5, 1), atol=1e-7, rtol=1e-7)
    end
    @testset "log1psq" begin
        rng = MersenneTwister(123456)
        @testset "Float64" begin
            for x in [-10.0, -5.0, -1.0, -eps(), 0.0, eps(), 1.0, 5.0, 10.0]
                adjoint_test(log1psq, randn(rng), x)
            end
        end
        @testset "Float32" begin
            for x in [-10f0, -5f0, -1f0, -eps(Float32), 0f0, eps(Float32), 1f0, 5f0, 10f0]
                adjoint_test(log1psq, randn(rng, Float32), x;
                    rtol=10_000 * eps(Float32),
                    atol=10_000 * eps(Float32),
                )
            end
        end
    end
    function test_log1pexp(T, rng, tol, xs)
        for x in xs
            adjoint_test(log1pexp, randn(rng, T), x;
                fdm=FiniteDifferences.Central(5, 1; eps=eps(T), adapt=2),
                rtol=tol,
                atol=tol,
            )
        end
    end
    @testset "log1pexp" begin
        @testset "Float64" begin
            @testset "x ∈ (-∞, 18.0)" begin
                test_log1pexp(Float64, MersenneTwister(123456), 1e5 * eps(),
                    [-1000.0, -50.0, -25.0, -10.0, 0.0, 10.0, 18.0 - eps()],
                )
            end
            @testset "x ∈ [18.0, 33.3)" begin
                test_log1pexp(Float64, MersenneTwister(123456), 1e5 * eps(),
                    [18.0, 18.0 + eps(), 33.3 - eps()],
                )
            end
            @testset "x ∈ [33.3, ∞)" begin
                test_log1pexp(Float64, MersenneTwister(123456), 1e5 * eps(),
                    [33.3, 33.3 + eps(), 100.0],
                )
            end
        end
        @testset "Float32" begin
            # @testset "x ∈ (-∞, 9f0)" begin
            #     test_log1pexp(Float32, MersenneTwister(123456), 10_000 * eps(Float32),
            #         [-1000f0, -50f0, -25f0, -10f0, 0f0, 5f0, 9f0 - eps(Float32)],
            #     )
            # end
            # @testset "x ∈ [9f0, 16f0)" begin
            #     test_log1pexp(Float32, MersenneTwister(123456), 10_000 * eps(Float32),
            #         [9f0, 9f0 + eps(Float32), 16f0 - eps(Float32)],
            #     )
            # end
            # @testset "x ∈ [16f0, ∞)" begin
            #     test_log1pexp(Float32, MersenneTwister(123456), 10_000 * eps(Float32),
            #         [16f0, 16f0 + eps(Float32), 100f0],
            #     )
            # end
        end
    end
    @testset "logsumexp" begin
        rng = MersenneTwister(123456)
        @testset "Float64" begin
            adjoint_test(logsumexp, randn(rng), randn(rng, 1))
            adjoint_test(logsumexp, randn(rng), randn(rng, 1, 1))
            adjoint_test(logsumexp, randn(rng), randn(rng, 3))
            adjoint_test(logsumexp, randn(rng), randn(rng, 3, 4, 5))
        end
        @testset "Float32" begin
            adjoint_test(logsumexp, randn(rng, Float32), randn(rng, Float32, 1);
                atol=1000 * eps(Float32), rtol=1000 * eps(Float32))
            adjoint_test(logsumexp, randn(rng, Float32), randn(rng, Float32, 1, 1);
                atol=1000 * eps(Float32), rtol=1000 * eps(Float32))
            adjoint_test(logsumexp, randn(rng, Float32), randn(rng, Float32, 2);
                atol=10_000 * eps(Float32), rtol=10_000 * eps(Float32))
            adjoint_test(logsumexp, randn(rng, Float32), randn(rng, Float32, 1, 2);
                atol=10_000 * eps(Float32), rtol=10_000 * eps(Float32))
        end
    end
end
