using Stheno: pairwise, pairwise_fallback

@testset "generic" begin

    # # Test utility.
    # let
    #     x, X = randn(5), randn(5, 3)
    #     @test nobs(x) == 5 && nfeatures(x) == 1 && getobs(x, 1) == x[1]
    #     @test nobs(X) == 3 && nfeatures(X) == 5 && getobs(X, 1) == X[:, 1]
    #     @test nobs([x, X]) == nobs(x) + nobs(X)

    #     @test getobs(X, 1:3) === X && getobs(X, :) === X
    #     @test getobs(x, 1:5) === x && getobs(x, :) === x

    #     @test eachobs(x) == eachindex(x)
    #     @test eachobs(X) == 1:3
    #     @test eachobs([x, X]) == [eachobs(x), eachobs(X)]
    # end

    # # Test unary_obswise.
    # let
    #     rng, P, D = MersenneTwister(123456), 3, 4
    #     X, x = randn(rng, D, P), randn(rng, P)
    #     foo_generic = x->sum(abs2, x)

    #     @test unary_obswise_fallback(foo_generic, X) ==
    #         [foo_generic(X[:, 1]), foo_generic(X[:, 2]), foo_generic(X[:, 3])]
    #     @test unary_obswise_fallback(foo_generic, x) ==
    #         [foo_generic(x[1]), foo_generic(x[2]), foo_generic(x[3])]

    #     @test unary_obswise(foo_generic, x) == unary_obswise_fallback(foo_generic, x)
    #     @test unary_obswise(foo_generic, X) == unary_obswise_fallback(foo_generic, X)
    # end

end
