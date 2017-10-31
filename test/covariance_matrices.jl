@testset "strided_covmat" begin

    # Test strided matrix functionality.
    let
        import Stheno.StridedPDMatrix
        rng = MersenneTwister(123456)
        A = randn(rng, 5, 5)
        K_ = A.'A + UniformScaling(1e-6)
        K = StridedPDMatrix(chol(K_))
        x = randn(rng, 5)

        # Test invariances.
        @test maximum(abs.(full(K) - K_)) < 1e-10 # Loss of accuracy > machine-ϵ.
        @test x.' * (K_ \ x) ≈ invquad(K, x)
        @test logdet(K) ≈ logdet(K_)

        @test size(K) == size(K_)
        @test size(K, 1) == size(K_, 1)
        @test size(K, 2) == size(K_, 2)

        @test K == K
        @test chol(K) == chol(K_)
    end

    # Test joint covariance matrix construction.
    let
        rng = MersenneTwister(123456)
        P, Q = 3, 2
        x, y = randn(rng, P), randn(rng, Q)

        gpc = GPC()
        f1, f2 = GP(x->0.0, EQ(), gpc), GP(x->0.0, RQ(1.0), gpc)
        K1, K2 = kernel(f1).(x, x'), kernel(f2).(y, y')
        @test K1 ≈ full(cov([f1(x)]))
        @test K2 ≈ full(cov([f2(y)]))

        K_manual = vcat(hcat(K1, zeros(P, Q)), hcat(zeros(Q, P), K2))
        @test K_manual ≈ full(cov([f1(x), f2(y)]))
    end

    # Test joint cross-covariance matrix construction.
    let
        rng = MersenneTwister(123456)
        P1, P2, Ps1, Ps2 = 3, 2, 4, 5
        x1, x2, xs1, xs2 = randn.(rng, [P1, P2, Ps1, Ps2])

        gpc = GPC()
        f1, f2 = GP(x->0.0, EQ(), gpc), GP(x->0.0, RQ(1.0), gpc)

        K11, K12 = kernel(f1, f1).(xs1, x1'), kernel(f1, f2).(xs1, x2')
        K21, K22 = kernel(f2, f1).(xs2, x1'), kernel(f2, f2).(xs2, x2')

        K_manual = vcat(hcat(K11, K12), hcat(K21, K22))
        @test K_manual == full(cov([f1(xs1), f2(xs2)], [f1(x1), f2(x2)]))
    end
end
