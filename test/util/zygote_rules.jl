@timedtestset "zygote_rules" begin
    @timedtestset "Cholesky (getproperty)" begin
        rng, N = MersenneTwister(123456), 5
        A = randn(rng, N, N)
        S = A' * A + 1e-6I
        C = cholesky(S)

        # Check that non-differentiable ops run forwards and have `nothing` gradients.
        _, back = Zygote.pullback(C->C.info, C)
        @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)
        _, back = Zygote.pullback(C->C.uplo, C)
        @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)

        # Unit test retrieving the factors.
        @test_throws ErrorException Zygote.pullback(C->C.factors, C)

        # Test getproperty.
        adjoint_test(A->Cholesky(A, :U, 0).U, randn(rng, N, N), A)
        adjoint_test(A->Cholesky(A, :U, 0).L, randn(rng, N, N), A)
        adjoint_test(A->Cholesky(A, :L, 0).U, randn(rng, N, N), A)
        adjoint_test(A->Cholesky(A, :L, 0).L, randn(rng, N, N), A)
    end
    @timedtestset "Diagonal" begin
        rng, N = MersenneTwister(123456), 11
        adjoint_test(Diagonal, rand(rng, N, N), randn(rng, N))
        adjoint_test(x->Diagonal(x).diag, randn(rng, N), randn(rng, N))
    end
    @timedtestset "broadcast" begin
        @timedtestset "exp" begin
            rng, N = MersenneTwister(123456), 11
            adjoint_test(x->exp.(x), randn(rng, N), randn(rng, N))
        end
        @timedtestset "-" begin
            rng, N = MersenneTwister(123456), 11
            adjoint_test(x->.-x, randn(rng, N), randn(rng, N))
        end
    end
    @timedtestset "ldiv(::Diagonal, ::Matrix)" begin
        rng, P, Q = MersenneTwister(123456), 13, 15
        Ȳ = randn(rng, P, Q)
        d = randn(rng, P)
        X = randn(rng, P, Q)
        adjoint_test((d, X)->Diagonal(d) \ X, Ȳ, d, X)
    end
end
