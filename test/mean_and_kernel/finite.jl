@testset "finite" begin

    # Test fully finite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1, k2, k3 = Finite(EQ(), x), Finite(EQ(), y), Finite(RQ(1.0), x)

        # Check equality works as expected
        @test k1 == k1
        @test k1 != k2
        @test k1 != k3
        @test k2 == k2
        @test k2 != k3
        @test k3 == k3

        # Check the correct things are evaluated.
        @test k1.(1:N, (1:N)') == EQ().(x, x')
        @test k2.(1:N, (1:N)') == EQ().(y, y')
        @test k3.(1:N, (1:N)') == RQ(1.0).(x, x')

        @test size(k1) == (N, N)
        @test size(k1, 1) == N
        @test size(k1, 2) == N

        if check_mem

            # Memory performance tests.
            @test memory(@benchmark Finite(EQ(), $x) seconds=0.1) == 32
            @test memory(@benchmark Finite(EQ(), $y) seconds=0.1) == 32
            @test memory(@benchmark Finite(RQ(1.0), $x) seconds=0.1) == 32

            @test memory(@benchmark $k1(1, 2) seconds=0.1) == 0
        end
    end

    # Test left-finite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1, k2, k3 = LhsFinite(EQ(), x), LhsFinite(EQ(), y), LhsFinite(RQ(1.0), x)

        # Check equality works as expected
        @test k1 == k1
        @test k1 != k2
        @test k1 != k3
        @test k2 == k2
        @test k2 != k3
        @test k3 == k3

        # Check the correct things are evaluated.
        @test k1.(1:N, x') == EQ().(x, x')
        @test k2.(1:N, y') == EQ().(y, y')
        @test k3.(1:N, x') == RQ(1.0).(x, x')

        if check_mem

            # Memory performance tests.
            @test memory(@benchmark LhsFinite(EQ(), $x) seconds=0.1) == 16
            @test memory(@benchmark LhsFinite(EQ(), $y) seconds=0.1) == 16
            @test memory(@benchmark LhsFinite(RQ(1.0), $x) seconds=0.1) == 32

            @test memory(@benchmark $k1(1, 2) seconds=0.1) == 0
        end
    end

    # Test right-finite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1, k2, k3 = RhsFinite(EQ(), x), RhsFinite(EQ(), y), RhsFinite(RQ(1.0), x)

        # Check equality works as expected
        @test k1 == k1
        @test k1 != k2
        @test k1 != k3
        @test k2 == k2
        @test k2 != k3
        @test k3 == k3

        # Check the correct things are evaluated.
        @test k1.(x, (1:N)') == EQ().(x, x')
        @test k2.(y, (1:N)') == EQ().(y, y')
        @test k3.(x, (1:N)') == RQ(1.0).(x, x')

        if check_mem

            # Memory performance tests.
            @test memory(@benchmark RhsFinite(EQ(), $x) seconds=0.1) == 16
            @test memory(@benchmark RhsFinite(EQ(), $y) seconds=0.1) == 16
            @test memory(@benchmark RhsFinite(RQ(1.0), $x) seconds=0.1) == 32

            @test memory(@benchmark $k1(1, 2) seconds=0.1) == 0
        end
    end

    # Test that we can nest Finite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1 = Finite(EQ(), x)

        # Check nesting from start.
        k2 = Finite(k1, 1:N-1)
        @test k2.(1:N-1, (1:N-1)') == k1.(1:N-1, (1:N-1)')

        # Check nesting from 2nd position.
        k3 = Finite(k1, 2:N)
        @test k3.(1:N-1, (1:N-1)') == k1.(2:N, (2:N)')
    end

    # Nest LhsFinite kernels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1 = LhsFinite(EQ(), x)

        # Nest into Finite.
        k2 = Finite(k1, 1:N, y)
        @test k2.(1:N, (1:N)') == k1.(1:N, y')

        # Nest into LhsFinite.
        k3 = LhsFinite(k1, 1:N-1)
        @test k3.(1:N-1, y') == k1.(1:N-1, y')
    end

    # Nest RhsFinite kenrels.
    let N = 5, rng = MersenneTwister(123456)

        # Construct kernels.
        x, y = randn(rng, N), randn(rng, N)
        k1 = RhsFinite(EQ(), y)

        # Nest into Finite.
        k2 = Finite(k1, x, 1:N)
        @test k2.(1:N, (1:N)') == k1.(x, (1:N)')

        # Nest into RhsFinite.
        k3 = RhsFinite(k1, 2:N)
        @test k3.(x, (1:N-1)') == k1.(x, (2:N)')
    end

    # Construct Finite from LhsFinite and RhsFinite and test for consistency.
    let N = 5, rng = MersenneTwister(123456)

        x, y = randn(rng, N), randn(rng, N)
        k_lhs, k_rhs = LhsFinite(EQ(), x), RhsFinite(EQ(), y)
        k_fin_lhs, k_fin_rhs = Finite(k_lhs, y), Finite(k_rhs, x)

        @test k_fin_lhs.(1:N, (1:N)') == k_lhs.(1:N, y')
        @test k_fin_rhs.(1:N, (1:N)') == k_rhs.(x, (1:N)')
    end
end
