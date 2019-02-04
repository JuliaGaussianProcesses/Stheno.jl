using FDM, Zygote, Distances, Random, LinearAlgebra, FillArrays, ToeplitzMatrices, StatsFuns

@testset "zygote_rules" begin
    @testset "FillArrays" begin
        rng, N = MersenneTwister(123456), 10
        x, y = randn(rng), randn(rng)
        @test Zygote.gradient(x->sum(Fill(x, N)), x)[1] == N
        @test Zygote.gradient(x->sum(Fill(x, N, 3, 4)), x)[1] == N * 3 * 4
        @test Zygote.gradient((x, y)->sum(Fill(x, N)), x, y) == (N, nothing)

        let
            out, back = Zygote.forward(sum, Fill(x, N))
            @test back(nothing) isa Nothing
        end

        # Test unary broadcasting gradients.
        out, back = Zygote.forward(x->exp.(x), Fill(x, N))
        @test out isa Fill
        @test out == Fill(exp(x), N)
        @test back(Ones(N))[1] isa Fill
        @test back(Ones(N))[1] == Ones(N) .* exp(x)
        @test back(ones(N))[1] isa Vector
        @test back(ones(N))[1] == ones(N) .* exp(x)
        adjoint_test(x->exp.(Fill(3x, N)), ones(N), x)
    end

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

    @testset "cholesky (SymmetricToeplitz)" begin
        rng, N = MersenneTwister(123456), 10
        x = pairwise(eq(), range(-3.0, stop=3.0, length=N))[:, 1]
        x[1] += 0.1 # ensure positive definite-ness under minor perturbations.
        adjoint_test(
            x->cholesky(SymmetricToeplitz(x)).U,
            randn(rng, N, N),
            x;
            rtol=1e-6,
            atol=1e-6,
        )
    end

    @testset "Diagonal" begin
        rng, N = MersenneTwister(123456), 11
        adjoint_test(Diagonal, rand(rng, N, N), randn(rng, N))
        adjoint_test(x->Diagonal(x).diag, randn(rng, N), randn(rng, N))
    end

    function test_log1pexp(T, rng, tol, xs)
        for x in xs
            adjoint_test(log1pexp, randn(rng, T), x;
                fdm=central_fdm(5, 1; ε=eps(T)),
                rtol=tol,
                atol=tol,
            )
        end
    end

    @testset "log1pexp (Float64)" begin
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

    @testset "log1pexp (Float32)" begin
        @testset "x ∈ (-∞, 9f0)" begin
            test_log1pexp(Float32, MersenneTwister(123456), 1000 * eps(Float32),
                [-1000f0, -50f0, -25f0, -10f0, 0f0, 5f0, 9f0 - eps(Float32)],
            )
        end
        @testset "x ∈ [9f0, 16f0)" begin
            test_log1pexp(Float32, MersenneTwister(123456), 1000 * eps(Float32),
                [9f0, 9f0 + eps(Float32), 16f0 - eps(Float32)],
            )
        end
        @testset "x ∈ [16f0, ∞)" begin
            test_log1pexp(Float32, MersenneTwister(123456), 1000 * eps(Float32),
                [16f0, 16f0 + eps(Float32), 100f0],
            )
        end
    end
end
