@timedtestset "zygote_rules" begin
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
