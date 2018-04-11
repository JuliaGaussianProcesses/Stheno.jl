@testset "cat" begin

    # Test CatMean.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        X1, X2 = randn(rng, N, D), randn(rng, N′, D)
        μ1, μ2 = FiniteMean(ConstantMean(1.0), X1), FiniteMean(ZeroMean{Float64}(), X2)
        μ = CatMean([μ1, μ2])
        @test length(μ) == length(μ1) + length(μ2)
        @test mean(μ) == vcat(mean(μ1), mean(μ2))
        @test CatMean(μ1, μ2) == μ

        μ3, μ4 = ConstantMean(1.0), ZeroMean{Float64}()
        μ′ = CatMean(μ3, μ4)
        @test mean(μ) == mean(μ′, BlockMatrix([X1, X2], 2, 1))
    end

    # Test CatCrossKernel.
    let
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 2, 7, 8
        X1, X2 = randn(rng, N1, D), randn(rng, N2, D)
        X1′, X2′ = randn(rng, N1′, D), randn(rng, N2′, D)
        k11 = FiniteCrossKernel(EQ(), X1, X1′)
        k12 = FiniteCrossKernel(ZeroKernel{Float64}(), X1, X2′)
        k21 = FiniteCrossKernel(ZeroKernel{Float64}(), X2, X1′)
        k22 = FiniteCrossKernel(EQ(), X2, X2′)
        k = CatCrossKernel([k11 k12; k21 k22])
        @test size(k) == (N1 + N2, N1′ + N2′)
        @test size(k, 1) == size(k)[1]
        @test size(k, 2) == size(k)[2]
        @test xcov(k) == vcat(hcat(xcov(k11), xcov(k12)), hcat(xcov(k21), xcov(k22)))

        v11, v12, v21, v22 = EQ(), ZeroKernel{Float64}(), ZeroKernel{Float64}(), EQ()
        k′ = CatCrossKernel([v11 v12; v21 v22])
        @test xcov(k) == xcov(k′, BlockMatrix([X1, X2], 2, 1), BlockMatrix([X1′, X2′], 2, 1))
    end

    # Test CatKernel.
    let
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 3, 4, 2
        X1, X2 = randn(rng, N1, D), randn(rng, N2, D)
        X1′, X2′ = randn(rng, N1′, D), randn(rng, N2′, D)

        # Construct CatKernel.
        k11, k22 = FiniteKernel(EQ(), X1), FiniteKernel(EQ(), X2)
        k12 = FiniteCrossKernel(ZeroKernel{Float64}(), X1, X2)
        ks_off = Matrix{FiniteCrossKernel}(undef, 2, 2)
        ks_off[1, 2] = k12
        k = CatKernel([k11, k22], ks_off)

        @test size(k) == (N1 + N2, N1 + N2)
        @test size(k, 1) == size(k)[1]
        @test size(k, 1) == size(k, 2)
        manual = vcat(hcat(Matrix(cov(k11)), xcov(k12)), hcat(zeros(N2, N1), Matrix(cov(k22))))
        @test cov(k) == Stheno.LazyPDMat(manual)

        # Compute cov for CatKernel with infinite kernels
        v11, v12, v22 = EQ(), ZeroKernel{Float64}(), EQ()
        ks_off′ = Matrix{Stheno.CrossKernel}(undef, 2, 2)
        ks_off′[1, 2] = v12
        k′ = CatKernel([v11, v22], ks_off′)
        @test cov(k) == cov(k′, BlockMatrix([X1, X2], 2, 1))

        # Compute xcov for CatKernel with infinite kernels.
        manual = vcat(hcat(xcov(v11, X1, X1′), xcov(v12, X1, X2′)),
                      hcat(xcov(v12, X2, X1′), xcov(v22, X2, X2′)),)
        @test xcov(k′, BlockMatrix([X1, X2], 2, 1), BlockMatrix([X1′, X2′], 2, 1)) == manual
    end
end
