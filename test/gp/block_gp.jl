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
        # @test eachindex(f_single) == [eachindex(f)]

        @test mean(f_single) isa BlockMean
        @test mean(f_single) == BlockMean([mean(f)])
        @test getblock(mean_vec(f_single), 1) == mean_vec(f)

        @test kernel(f_single) isa BlockKernel
        @test kernel(f_single).ks_diag == [kernel(f)]
        @test getblock(Stheno.unbox(cov(f_single)), 1, 1) == cov(f)

        @test xcov(f_single, f) isa BlockMatrix
        @test xcov(f_single, g) isa BlockMatrix
        @test xcov(f, f_single) isa BlockMatrix
        @test xcov(g, f_single) isa BlockMatrix
        @test size(xcov(f_single, f)) == (N, N)
        @test size(xcov(f_single, g)) == (N, N′)
        @test size(xcov(f, f_single)) == (N, N)
        @test size(xcov(g, f_single)) == (N′, N)
        @test xcov(f_single, f) == BlockMatrix([xcov(f, f)])
        @test xcov(f_single, g) == BlockMatrix([xcov(f, g)])
        @test xcov(f, f_single) == BlockMatrix([xcov(f, f)])
        @test xcov(g, f_single) == BlockMatrix([xcov(g, f)])

        @test length(rand(rng, f_single)) == length(f_single)
        @test size(rand(rng, f_single, 3)) == (length(f_single), 3)

        # Construct a GP over multiple processes.
        fs = BlockGP([f, g])
        @test length(fs) == length(f) + length(g)
        # @test eachindex(fs) == [eachindex(f), eachindex(g)]

        @test mean(fs) isa BlockMean
        @test mean(fs) == BlockMean([mean(f), mean(g)])
        @test mean_vec(fs) == BlockVector([mean_vec(f), mean_vec(g)])

        @test getblock(Stheno.unbox(cov(fs)), 1, 1) == cov(f)
        @test getblock(Stheno.unbox(cov(fs)), 2, 2) == cov(g)
        @test getblock(Stheno.unbox(cov(fs)), 1, 2) == xcov(f, g)
        @test getblock(Stheno.unbox(cov(fs)), 2, 1) == xcov(g, f)

        # k = Stheno.BlockCrossKernel(kernel.(f, permutedims(fs.fs)))
        # @show size(k.ks[1]), size(k.ks[2])
        # @show eachindex(k, 1), eachindex(k, 2)

        @test xcov(fs, f) isa BlockMatrix
        @test xcov(fs, g) isa BlockMatrix
        @test xcov(f, fs) isa BlockMatrix
        @test xcov(g, fs) isa BlockMatrix
        @test size(xcov(fs, f)) == (N + N′, N)
        @test size(xcov(fs, g)) == (N + N′, N′)
        @test size(xcov(f, fs)) == (N, N + N′)
        @test size(xcov(g, fs)) == (N′, N + N′)
        @test xcov(fs, f) == BlockMatrix(reshape([xcov(f, f), xcov(g, f)], 2, 1))

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
