using Random, LinearAlgebra
using Stheno: BlockGP, getblock, LazyPDMat
using Distributions: MvNormal, PDMat

@testset "block_gp" begin

    let
        rng, N, N′, D, gpc = MersenneTwister(123456), 3, 4, 2, GPC()
        X, X′ = ColsAreObs(rand(rng, D, N)), ColsAreObs(rand(rng, D, N′))
        f = GP(FiniteMean(ConstantMean(5.4), X), FiniteKernel(EQ(), X), gpc)
        g = GP(FiniteMean(ConstantMean(1.2), X′), FiniteKernel(EQ(), X′), gpc)
        y, z = rand(rng, f), rand(rng, g)

        # Construct a BlockGP over a single finite process (edge-case).
        f_single = BlockGP([f])
        @test length(f_single) == length(f)

        @test mean(f_single) isa BlockMean
        @test mean(f_single) == BlockMean([mean(f)])
        @test getblock(mean(f_single), 1) == mean(f)

        @test kernel(f_single) isa BlockKernel
        @test kernel(f_single).ks_diag == [kernel(f)]
        @test getblock(Stheno.unbox(cov(f_single)), 1, 1) == cov(f)

        @test cov(f_single, f) isa BlockMatrix
        @test cov(f_single, g) isa BlockMatrix
        @test cov(f, f_single) isa BlockMatrix
        @test cov(g, f_single) isa BlockMatrix
        @test size(cov(f_single, f)) == (N, N)
        @test size(cov(f_single, g)) == (N, N′)
        @test size(cov(f, f_single)) == (N, N)
        @test size(cov(g, f_single)) == (N′, N)
        @test cov(f_single, f) == BlockMatrix([cov(f, f)])
        @test cov(f_single, g) == BlockMatrix([cov(f, g)])
        @test cov(f, f_single) == BlockMatrix([cov(f, f)])
        @test cov(g, f_single) == BlockMatrix([cov(g, f)])

        @test length(rand(rng, f_single)) == length(f_single)
        @test size(rand(rng, f_single, 3)) == (length(f_single), 3)

        # Construct a GP over multiple processes.
        fs = BlockGP([f, g])
        @test length(fs) == length(f) + length(g)
        # @test eachindex(fs) == [eachindex(f), eachindex(g)]

        @test mean(fs) isa BlockMean
        @test mean(fs) == BlockMean([mean(f), mean(g)])
        @test mean(fs) == BlockVector([mean(f), mean(g)])

        @test getblock(Stheno.unbox(cov(fs)), 1, 1) == cov(f)
        @test getblock(Stheno.unbox(cov(fs)), 2, 2) == cov(g)
        @test getblock(Stheno.unbox(cov(fs)), 1, 2) == cov(f, g)
        @test getblock(Stheno.unbox(cov(fs)), 2, 1) == cov(g, f)

        @test cov(fs, f) isa BlockMatrix
        @test cov(fs, g) isa BlockMatrix
        @test cov(f, fs) isa BlockMatrix
        @test cov(g, fs) isa BlockMatrix
        @test size(cov(fs, f)) == (N + N′, N)
        @test size(cov(fs, g)) == (N + N′, N′)
        @test size(cov(f, fs)) == (N, N + N′)
        @test size(cov(g, fs)) == (N′, N + N′)
        @test cov(fs, f) == BlockMatrix(reshape([cov(f, f), cov(g, f)], 2, 1))

        @test length(rand(rng, fs)) == length(fs)
        @test size(rand(rng, fs, 3)) == (length(fs), 3)
        @test sum(length.(rand(rng, [f, g]))) == length(f) + length(g)
        fs, gs = rand(rng, [f, g], 4)
        @test size(fs) == (length(f), 4) && size(gs) == (length(g), 4)

        # Check `logpdf` for two independent processes.
        joint, joint_obs = BlockGP([f, g]), BlockVector([y, z])
        @test logpdf(joint, joint_obs) ≈ logpdf(f, y) + logpdf(g, z)
        @test logpdf([f, g], [y, z]) == logpdf(joint, joint_obs)
    end
end
