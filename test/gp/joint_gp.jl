using Stheno: JointGP

@testset "joint_gp" begin

    using Distributions: MvNormal, PDMat
    let
        rng, N, N′, D, gpc = MersenneTwister(123456), 25, 10, 2, GPC()
        X, X′ = rand(rng, D, N), rand(rng, D, N′)
        f = GP(ConstantMean(1.0), EQ(), gpc)
        y, y′ = rand(rng, f(X)), rand(rng, f(X′))


        # Construct a JointGP over a single finite process (edge-case).
        f_single = JointGP([f(X)])
        @test length(f_single) == length(f(X))
        @test eachindex(f_single) == [eachindex(f(X))]

        @test mean(f_single) isa CatMean
        @test mean(f_single) == CatMean([mean(f(X))])
        @test getblock(mean_vec(f_single), 1) == mean_vec(f(X))

        @test kernel(f_single) isa CatKernel
        @test kernel(f_single).ks_diag == [kernel(f(X))]
        @test getblock(Stheno.unbox(cov(f_single)), 1, 1) == cov(f(X))

        @test length(rand(rng, f_single)) == length(f_single)
        @test size(rand(rng, f_single, 3)) == (length(f_single), 3)


        # Construct a GP over multiple processes.
        fs = JointGP([f(X), f(X′)])
        @test length(fs) == length(f(X)) + length(f(X′))
        @test eachindex(fs) == [eachindex(f(X)), eachindex(f(X′))]

        @test mean(fs) isa CatMean
        @test mean(fs) == CatMean([mean(f(X)), mean(f(X′))])
        @test mean_vec(fs) == BlockVector([mean_vec(f(X)), mean_vec(f(X′))])

        @test getblock(Stheno.unbox(cov(fs)), 1, 1) == cov(f(X))
        @test getblock(Stheno.unbox(cov(fs)), 2, 2) == cov(f(X′))
        @test getblock(Stheno.unbox(cov(fs)), 1, 2) == xcov(f(X), f(X′))
        @test getblock(Stheno.unbox(cov(fs)), 2, 1) == xcov(f(X′), f(X))

        @test length(rand(rng, fs)) == length(fs)
        @test size(rand(rng, fs, 3)) == (length(fs), 3)
        @test length(rand(rng, [f(X), f(X′)])) == length(f(X)) + length(f(X′))
        @test size(rand(rng, [f(X), f(X′)], 4)) == (length(f(X)) + length(f(X′)), 4)

        # Check `logpdf` for two independent processes.
        g = GP(EQ(), gpc)
        ĝ = rand(rng, g(X))
        joint, joint_obs = JointGP([f(X), g(X)]), BlockVector([y, ĝ])
        @test logpdf(joint, joint_obs) ≈ logpdf(f(X), y) + logpdf(g(X), ĝ)
        @test logpdf([f(X), g(X)], [y, ĝ]) == logpdf(joint, joint_obs)
    end
end
