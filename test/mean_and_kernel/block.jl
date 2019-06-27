using Stheno: ZeroMean, OneMean, ZeroKernel, BlockMean, BlockKernel, BlockData,
    BlockCrossKernel, map, pw, EQ, CustomMean

@testset "block" begin
    @testset "BlockMean" begin
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x1, x2 = randn(rng, N), randn(rng, N′)
        m1, m2 = ZeroMean(), CustomMean(sin)
        m = BlockMean([m1, m2])

        # Unary elementwise.
        @test ew(m, BlockData([x1, x2])) == vcat(ew(m1, x1), ew(m2, x2))
        adjoint_test(
            (x1, x2)->ew(BlockMean([m1, m2]), BlockData([x1, x2])),
            randn(rng, N + N′),
            x1, x2,
        )

        # Consistency tests.
        mean_function_tests(m, BlockData([x1, x2]))
        differentiable_mean_function_tests(m, randn(rng, N + N′), BlockData([x1, x2]))
    end
    @testset "BlockCrossKernel" begin
        @testset "block-block" begin
            rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 2, 7, 8
            X0, X0′ = randn(rng, N1), randn(rng, N2)
            X1, X1′ = randn(rng, N1′), randn(rng, N2′)
            X2, X2′ = randn(rng, N1), randn(rng, N2)
            k11, k12, k21, k22 =  EQ(), ZeroKernel(), ZeroKernel(), EQ()
            k = BlockCrossKernel([k11 k12; k21 k22])

            D, D′ = BlockData([X0, X0′]), BlockData([X2, X2′])

            # Binary elementwise.
            @test ew(k, D, D′) == vcat(ew(k11, X0, X2), ew(k22, X0′, X2′))
            adjoint_test(
                (x0, x0′, x2, x2′)->ew(k, BlockData([x0, x0′]), BlockData([x2, x2′])),
                randn(rng, N1 + N2),
                X0, X0′, X2, X2′,
            )

            # Binary pairwise.
            D, D′ = BlockData([X0, X1]), BlockData([X0′, X1′])
            row1 = hcat(pw(k11, X0, X0′), pw(k12, X0, X1′))
            row2 = hcat(pw(k21, X1, X0′), pw(k22, X1, X1′))
            K = vcat(row1, row2)
            @test pw(k, D, D′) == K
            adjoint_test(
                (x0, x0′, x1, x1′)->pw(k, BlockData([x0, x1]), BlockData([x0′, x1′])),
                randn(rng, N1 + N1′, N2 + N2′),
                X0, X0′, X1, X1′,
            )

            # Consistency tests.
            ȳ, Ȳ = randn(rng, N1 + N2), randn(rng, N1 + N2, N1′ + N2′)
            D0, D1, D2 = BlockData([X0, X0′]), BlockData([X2, X2′]), BlockData([X1, X1′])
            differentiable_cross_kernel_tests(k, ȳ, Ȳ, D0, D1, D2)
        end
        @testset "block-single" begin
            rng, N1, N2, N1′ = MersenneTwister(123456), 3, 5, 7
            x0, x0′ = randn(rng, N1), randn(rng, N2)
            x1, x1′ = randn(rng, N1), randn(rng, N2)
            x2 = randn(rng, N1′)

            k11, k21 = EQ(), EQ()
            k = BlockCrossKernel([k11; k21])

            # Test binary pairwise for block and non-block rhs argument.
            @test pw(k, BlockData([x0, x0′]), BlockData([x2])) ==
                pw(k, BlockData([x0, x0′]), x2)
            adjoint_test(
                (x0, x0′, x2)->pw(k, BlockData([x0, x0′]), x2),
                randn(rng, N1 + N2, N1′),
                x0, x0′, x2,
            )
        end
        @testset "single-block" begin
            rng, N1, N2, N1′, N2′ = MersenneTwister(123456), 3, 5, 7, 9
            x0 = randn(rng, N1)
            x1, x1′ = randn(rng, N1), randn(rng, N2)
            x2, x2′ = randn(rng, N1′), randn(rng, N2′)

            k11, k12 = EQ(), EQ()
            k = BlockCrossKernel(reshape([k11 k12], 1, :))

            # Test binary pairwise for block and non-block lhs argument.
            @test pw(k, BlockData([x0]), BlockData([x2, x2′])) ==
                pw(k, x0, BlockData([x2, x2′]))
            adjoint_test(
                (x0, x2, x2′)->pw(k, x0, BlockData([x2, x2′])),
                randn(rng, N1, N1′ + N2′),
                x0, x2, x2′,
            )
        end
    end
    @testset "BlockKernel" begin
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 3, 4, 2
        x0, x0′ = randn(rng, N1), randn(rng, N2)
        x1, x1′ = randn(rng, N1′), randn(rng, N2′)
        x2, x2′ = randn(rng, N1), randn(rng, N2)
        D0, D1, D2 = BlockData([x0, x0′]), BlockData([x1, x1′]), BlockData([x2, x2′])

        # Construct BlockKernel.
        k11, k12, k21, k22 =  EQ(), ZeroKernel(), ZeroKernel(), EQ()
        k = BlockKernel([k11 k12; k21 k22])

        # Binary elementwise.
        @test ew(k, D0, D2) == vcat(ew(k11, x0, x2), ew(k22, x0′, x2′))

        # Binary pairwise.
        D, D′ = BlockData([x0, x1]), BlockData([x0′, x1′])
        K = vcat(
            hcat(pw(k11, x0, x0′), pw(k12, x0, x1′)),
            hcat(pw(k21, x1, x0′), pw(k22, x1, x1′)),
        )
        @test pw(k, D, D′) == K

        # Unary elementwise.
        @test ew(k, D0) == vcat(ew(k11, x0), ew(k22, x0′))

        # Unary pairwise.
        row1 = hcat(pw(k11, x0), pw(k12, x0, x0′))
        row2 = hcat(zeros(N2, N1), pw(k22, x0′))
        @test pw(k, D0) == vcat(row1, row2)

        # Consistency tests.
        ȳ = randn(rng, N1 + N2)
        Ȳ = randn(rng, N1 + N2, N1′ + N2′)
        Ȳ_sq = randn(rng, N1 + N2, N1 + N2)
        differentiable_kernel_tests(k, ȳ, Ȳ, Ȳ_sq, D0, D2, D1)
    end
end
