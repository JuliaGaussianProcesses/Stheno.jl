@timedtestset "fillarrays" begin
    @timedtestset "Diagonal" begin
        rng, N = MersenneTwister(123456), 11
        d, Ȳ = randn(rng, N), randn(rng, N, N)
        adjoint_test(Diagonal, Ȳ, d)
    end
    @timedtestset "diag" begin
        @timedtestset "Matrix" begin
            rng, N = MersenneTwister(123456), 11
            X, ȳ = randn(rng, N, N), randn(rng, N)
            adjoint_test(diag, ȳ, X)
        end
        @timedtestset "Symmetric" begin
            rng, N = MersenneTwister(123456), 11
            X, ȳ = randn(rng, N, N), randn(rng, N)
            adjoint_test(diag, ȳ, Symmetric(X))
            adjoint_test(X->diag(Symmetric(X)), ȳ, X)
        end
        @timedtestset "Diagonal" begin
            rng, N = MersenneTwister(123456), 11
            d, ȳ = randn(rng, N), randn(rng, N)
            D = Diagonal(d)
            adjoint_test(diag, ȳ, D)
            adjoint_test(d->diag(Diagonal(d)), ȳ, D)
        end
        @timedtestset "Symmetric(Diagonal(...))" begin
            rng, N = MersenneTwister(123456), 11
            d, ȳ = randn(rng, N), randn(rng, N)
            adjoint_test(d->diag(Symmetric(Diagonal(d))), ȳ, d)
        end
        @timedtestset "Diagonal(::Fill)" begin
            rng, N = MersenneTwister(123456), 11
            x, ȳ = randn(rng), randn(rng, N)
            adjoint_test(x->diag(Diagonal(Fill(x, N))), ȳ, x)
        end
    end
    @timedtestset "cholesky(Diagonal(Fill(...)))" begin
        rng, N = MersenneTwister(123456), 4
        d = Fill(1.5, N)
        D = Diagonal(d)
        @test Matrix(cholesky(D).U) ≈ cholesky(Matrix(D)).U
        @test Zygote.gradient(D->sum(cholesky(D).U), D)[1] isa Diagonal{<:Real, <:Fill}
        @test Zygote.gradient(d->sum(cholesky(Diagonal(d)).U), d)[1] isa Fill
        adjoint_test(x->cholesky(Diagonal(Fill(x, N))).U, randn(rng, N, N), 1.5)
    end
    @timedtestset "cholesky(Symmetric(Diagonal(Fill(...))))" begin
        d = Fill(1.5, 4)
        S = Symmetric(Diagonal(d))
        rng, N = MersenneTwister(123456), 11
        @test Matrix(cholesky(S).U) ≈ cholesky(Matrix(S)).U
        @test Zygote.gradient(d->sum(cholesky(Symmetric(Diagonal(d))).U), d)[1] isa Fill
        adjoint_test(x->cholesky(Symmetric(Diagonal(Fill(x, N)))).U, randn(rng, N, N), 1.5)
    end
end
