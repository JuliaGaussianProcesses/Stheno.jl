@testset "strided_covmat" begin

    # Test strided matrix functionality.
    let
        import Stheno.StridedPDMatrix
        rng = MersenneTwister(123456)
        A = randn(rng, 5, 5)
        K_ = Transpose(A) * A + UniformScaling(1e-6)
        K = StridedPDMatrix(K_)
        x = randn(rng, 5)

        # Test invariances.
        @test maximum(abs.(full(K) - K_)) < 1e-10 # Loss of accuracy > machine-ϵ.
        @test Transpose(x) * (K_ \ x) ≈ invquad(K, x)
        @test logdet(K) ≈ logdet(K_)

        @test size(K) == size(K_)
        @test size(K, 1) == size(K_, 1)
        @test size(K, 2) == size(K_, 2)

        @test K == K
        @test chol(K) == chol(K_ + Stheno.__ϵ * I)
    end
end
