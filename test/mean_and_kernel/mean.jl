using Stheno: CustomMean, ZeroMean, ConstantMean, EmpiricalMean

@testset "mean" begin

    # Test fallbacks.
    let
        @test_throws ErrorException eachindex(CustomMean(sin))
        @test_throws AssertionError AbstractVector(CustomMean(cos))

        m = randn(5)
        @test AbstractVector(EmpiricalMean(m)) == m
    end

    # Test CustomMean function.
    let
        rng, N, D = MersenneTwister(123456), 11, 2
        X, x = ColsAreObs(randn(rng, D, N)), randn(rng, N)
        foo_mean = x->sum(abs, x)
        f = CustomMean(foo_mean)

        @test f(X[1]) == foo_mean(X[1])

        mean_function_tests(f, x)
        mean_function_tests(f, X)
        mean_function_tests(f, BlockData([x, X]))

        @test isinf(length(f))

        # Check that shorthand for block-wise application of mean function works.
        @test map(f, BlockData([x, X])) isa BlockVector
        @test map(f, BlockData([x, X])) == map(f, BlockData([x, X]))
    end

    # Test ZeroMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        f = ZeroMean{Float64}()

        @test f(randn(rng)) === zero(Float64)
        @test f == ZeroMean{Float32}()

        mean_function_tests(f, x)
        mean_function_tests(f, X)
        mean_function_tests(f, BlockData([x, X]))

        @test isinf(length(f))

        c = ConstantMean(5.0)
        @test f + f === f
        @test f + c == c
        @test c + f == c
        @test f * f === f
        @test f * c === f
        @test c * f === f
    end

    # Test ConstantMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = ColsAreObs(randn(rng, D, P)), randn(rng, P)
        c = randn(rng)
        f = ConstantMean(c)

        @test ConstantMean(1.0) == ConstantMean(1)
        @test ConstantMean(1.0) ≠ ConstantMean(2)
        @test f(randn(rng)) == c

        mean_function_tests(f, x)
        mean_function_tests(f, X)
        mean_function_tests(f, BlockData([x, X]))

        @test isinf(length(f))
    end

    # Test EmpiricalMean.
    let
        rng, N, D = MersenneTwister(123456), 11, 2
        x = 1:N
        m = randn(rng, N)
        μ = EmpiricalMean(m)

        @test length(μ) == length(m)
        @test eachindex(μ) == eachindex(m)
        @test μ == μ

        mean_function_tests(μ, x)
        mean_function_tests(μ, BlockData([x, x]))

        @test map(μ, :) == map(μ, eachindex(μ))
        @test map(μ, :) == m

    end
end
