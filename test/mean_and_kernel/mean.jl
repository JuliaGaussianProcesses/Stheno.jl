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
        X, x = DataSet(randn(rng, D, N)), DataSet(randn(rng, N))
        foo_mean = x->sum(abs, x)
        f = CustomMean(foo_mean)

        @test f(X[1]) == foo_mean(X[1])

        mean_function_tests(f, x)
        mean_function_tests(f, X)
        mean_function_tests(f, BlockData([x, X]))

        @test isinf(length(f))

        # Check that shorthand for block-wise application of mean function works.
        @test map(f, [x, X]) isa BlockVector
        @test map(f, [x, X]) == map(f, BlockData([x, X]))
    end

    # Test ZeroMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = DataSet(randn(rng, D, P)), DataSet(randn(rng, P))
        f = ZeroMean{Float64}()

        @test f(randn(rng)) === zero(Float64)
        @test f == ZeroMean{Float32}()

        mean_function_tests(f, x)
        mean_function_tests(f, X)
        mean_function_tests(f, BlockData([x, X]))

        @test isinf(length(f))
    end

    # Test ConstantMean.
    let
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        X, x = DataSet(randn(rng, D, P)), DataSet(randn(rng, P))
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
        Dx = DataSet(1:N)
        m = randn(rng, N)
        μ = EmpiricalMean(m)

        @test length(μ) == length(m)
        @test eachindex(μ) == eachindex(m)
        @test μ == μ

        mean_function_tests(μ, Dx)
        mean_function_tests(μ, BlockData([Dx, Dx]))
    end
end
