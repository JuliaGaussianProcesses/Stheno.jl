@testset "finite" begin

    # Test fully finite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1, k2, k3 = FullFinite(EQ(), x), FullFinite(EQ(), y), FullFinite(RQ(1.0), x)

        # Check equality works as expected
        @test k1 == k1
        @test k1 != k2
        @test k1 != k3
        @test k2 == k2
        @test k2 != k3
        @test k3 == k3

        # Check the correct things are evaluated.
        for p in 1:N, q in 1:N
            @test k1(p, q) == EQ()(x[p], x[q])
            @test k2(p, q) == EQ()(y[p], y[q])
            @test k3(p, q) == RQ(1.0)(x[p], x[q])
        end

        # Memory performance tests.
        @test memory(@benchmark FullFinite(EQ(), $x) seconds=0.1) == 32
        @test memory(@benchmark FullFinite(EQ(), $y) seconds=0.1) == 32
        @test memory(@benchmark FullFinite(RQ(1.0), $x) seconds=0.1) == 32

        @test memory(@benchmark $k1(1, 2) seconds=0.1) == 0
    end

    # Test left-finite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1, k2, k3 = LeftFinite(EQ(), x), LeftFinite(EQ(), y), LeftFinite(RQ(1.0), x)

        # Check equality works as expected
        @test k1 == k1
        @test k1 != k2
        @test k1 != k3
        @test k2 == k2
        @test k2 != k3
        @test k3 == k3

        # Check the correct things are evaluated.
        for p in 1:N, q in 1:N
            @test k1(p, x[q]) == EQ()(x[p], x[q])
            @test k2(p, y[q]) == EQ()(y[p], y[q])
            @test k3(p, x[q]) == RQ(1.0)(x[p], x[q])
        end

        # Memory performance tests.
        @test memory(@benchmark LeftFinite(EQ(), $x) seconds=0.1) == 16
        @test memory(@benchmark LeftFinite(EQ(), $y) seconds=0.1) == 16
        @test memory(@benchmark LeftFinite(RQ(1.0), $x) seconds=0.1) == 32

        @test memory(@benchmark $k1(1, 2) seconds=0.1) == 0
    end

    # Test right-finite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1, k2, k3 = RightFinite(EQ(), x), RightFinite(EQ(), y), RightFinite(RQ(1.0), x)

        # Check equality works as expected
        @test k1 == k1
        @test k1 != k2
        @test k1 != k3
        @test k2 == k2
        @test k2 != k3
        @test k3 == k3

        # Check the correct things are evaluated.
        for p in 1:N, q in 1:N
            @test k1(x[p], q) == EQ()(x[p], x[q])
            @test k2(y[p], q) == EQ()(y[p], y[q])
            @test k3(x[p], q) == RQ(1.0)(x[p], x[q])
        end

        # Memory performance tests.
        @test memory(@benchmark RightFinite(EQ(), $x) seconds=0.1) == 16
        @test memory(@benchmark RightFinite(EQ(), $y) seconds=0.1) == 16
        @test memory(@benchmark RightFinite(RQ(1.0), $x) seconds=0.1) == 32

        @test memory(@benchmark $k1(1, 2) seconds=0.1) == 0
    end
end
