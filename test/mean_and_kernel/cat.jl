using Stheno: CatMean, CatCrossKernel, CatKernel, ConstantMean, ZeroMean, nobs, getobs
using FillArrays

@testset "cat" begin

    # Test `nobs` and `getobs` for nested data.
    let
        rng, N1, N2, N3 = MersenneTwister(123456), 4, 1, 5
        x1, x2, x3 = randn(rng, N1), randn(rng, N2), randn(rng, N3)
        x, xcat = [x1, x2, x3], vcat(x1, x2, x3)

        @test nobs(x) == length(xcat)
        ids = [1, 1, 1, 1, 2, 3, 3, 3, 3, 3]
        @test all(map(n->getobs(x, n)[1], eachindex(xcat)) == ids)
        @test all(map(n->getobs(x, n)[2], eachindex(xcat)) == xcat)
    end

    # Test CatMean.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x1, x2 = randn(rng, N), randn(rng, N′)
        X1, X2 = randn(rng, D, N), randn(rng, D, N′)
        μ1, μ2 = ConstantMean(1.0), ZeroMean{Float64}()
        μ = CatMean([μ1, μ2])

        @test μ == CatMean(μ1, μ2)
        @test unary_obswise(μ, [X1, X2]) == vcat(unary_obswise(μ1, X1), unary_obswise(μ2, X2))
        @test unary_obswise(μ, [x1, x2]) == vcat(unary_obswise(μ1, x1), unary_obswise(μ2, x2))

        mean_function_tests(μ, [X1, X2])
        mean_function_tests(μ, [x1, x2])
    end

    # Test CatCrossKernel.
    let
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 2, 7, 8
        X0, X0′ = randn(rng, D, N1), randn(rng, D, N2)
        X1, X1′ = randn(rng, D, N1′), randn(rng, D, N2′)
        X2, X2′ = randn(rng, D, N1), randn(rng, D, N2)
        k11, k12, k21, k22 =  EQ(), ZeroKernel{Float64}(), ZeroKernel{Float64}(), EQ()
        k = CatCrossKernel([k11 k12; k21 k22])

        @test size(k) == (Inf, Inf)
        @test size(k, 1) == Inf
        @test size(k, 2) == Inf

        row1 = hcat(pairwise(k11, X0, X0′), pairwise(k12, X0, X1′))
        row2 = hcat(pairwise(k21, X1, X0′), pairwise(k22, X1, X1′))
        @test pairwise(k, [X0, X1], [X0′, X1′]) == vcat(row1, row2)

        cross_kernel_tests(k, [X0, X0′], [X2, X2′], [X1, X1′])
    end

    # Test CatKernel.
    let
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 3, 4, 2
        X0, X0′ = randn(rng, D, N1), randn(rng, D, N2)
        X1, X1′ = randn(rng, D, N1′), randn(rng, D, N2′)
        X2, X2′ = randn(rng, D, N1), randn(rng, D, N2)

        # Construct CatKernel.
        k11, k22, k12 = EQ(), EQ(), ZeroKernel{Float64}()
        ks_off = Matrix{Stheno.CrossKernel}(2, 2)
        ks_off[1, 2] = k12
        k = CatKernel([k11, k22], ks_off)

        @test size(k) == (Inf, Inf)
        @test size(k, 1) == Inf
        @test size(k, 2) == Inf

        row1 = hcat(pairwise(k11, X0), pairwise(k12, X0, X0′))
        row2 = hcat(Zeros{Float64}(N2, N1), pairwise(k22, X0′))
        @test pairwise(k, [X0, X0′]) == vcat(row1, row2)

        # Compute xcov for CatKernel with infinite kernels.
        manual = vcat(hcat(pairwise(k11, X1, X1′), pairwise(k12, X1, X2′)),
                      hcat(pairwise(k12, X2, X1′), pairwise(k22, X2, X2′)),)
        @test pairwise(k, [X1, X2], [X1′, X2′]) == manual

        kernel_tests(k, [X0, X0′], [X2, X2′], [X1, X1′])

        # _generic_kernel_tests(k, [X1, X2], [X1′, X2′])
    end
end
