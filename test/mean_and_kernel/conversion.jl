using Stheno: EmpiricalMean, EmpiricalKernel
using FillArrays: Zeros, Fill

@testset "conversion" begin

    # Mean function conversion.
    let
        rng, N = MersenneTwister(123456), 11
        x = randn(rng, N)

        # Check that base mean functions convert properly when made finite.
        for (μ, T) in [(CustomMean(sin), Vector{Float64}),
                       (ZeroMean{Float64}(), Zeros{Float64}),
                       (ConstantMean(randn(rng)), Fill)]
            μ_finite = FiniteMean(μ, x)
            @test AbstractVector(μ_finite) isa T
            @test AbstractVector(μ_finite) == map(μ, x)
        end

        # Test `EmpiricalMean` converts correctly.
        @test AbstractVector(EmpiricalMean(x)) === x

        # Check that `BlockMean`s behave well in both shallow finite scenarios.
        x1, x2 = randn(rng, 11), randn(rng, 7)
        μ1, μ2 = ConstantMean(randn(rng)), ZeroMean{Float64}()
        μ1f, μ2f = FiniteMean(μ1, x1), FiniteMean(μ2, x2)
        bf_μ = FiniteMean(BlockMean([μ1, μ2]), BlockData([x1, x2]))
        b_μf = BlockMean([μ1f, μ2f])

        @test AbstractVector(bf_μ) isa BlockVector
        @test AbstractVector(bf_μ) == vcat(AbstractVector(μ1f), AbstractVector(μ2f))

        @test AbstractVector(b_μf) isa BlockVector
        @test AbstractVector(bf_μ) == vcat(AbstractVector(μ1f), AbstractVector(μ2f))
    end

    # Kernel conversion.
    let
        rng, N = MersenneTwister(123456), 11
        x = randn(rng, N)

        # Test that base kernel functions convert properly when made finite.
        for (k, T) in [(ZeroKernel{Float64}(), Zeros),
                       (ConstantKernel(5.0), Fill),
                       (EQ(), Matrix{Float64}),
                       (Linear(0.0), Matrix{Float64}),
                       (Noise(1.0), Diagonal)]
            k_finite = FiniteKernel(k, x)
            @test AbstractMatrix(k_finite) isa LazyPDMat{Float64, <:T}
            @test AbstractMatrix(k_finite) == pairwise(k, x)
        end

        A = randn(5, 5)
        Σ = LazyPDMat(A' * A + eps() * I)
        @test AbstractMatrix(EmpiricalKernel(Σ)) === Σ

        # Check that `BlockKernel`s play nicely when made finite.
        x1, x2 = randn(rng, 11), randn(rng, 7)
        k1, k2, k12 = EQ(), Linear(1.0), ZeroKernel{Float64}()
        ks_off = Matrix{Stheno.CrossKernel}(undef, 2, 2)
        ks_off[1, 2] = k12
        bk = BlockKernel([k1, k2], ks_off)
        bf_k = FiniteKernel(bk, BlockData([x1, x2]))

        @test AbstractMatrix(bf_k) isa LazyPDMat{Float64, <:Symmetric{Float64, <:BlockMatrix}}
        @test AbstractMatrix(bf_k) == pairwise(bk, BlockData([x1, x2]))
    end

    # CrossKernel conversions.
    let
        # Check that CrossKernels play nicely when made finite.
        rng = MersenneTwister(123456)
        x1, x2, x3, x4 = randn(rng, 11), randn(rng, 7), randn(rng, 3), randn(rng, 13)
        k1, k2, k12, k21 = EQ(), EQ(), EQ(), EQ()
        bk = BlockCrossKernel([k1 k12; k21 k2])
        bf_k = FiniteCrossKernel(bk, BlockData([x1, x2]), BlockData([x3, x4]))

        @test AbstractMatrix(bf_k) isa BlockMatrix
        @test AbstractMatrix(bf_k) == pairwise(bk, BlockData([x1, x2]), BlockData([x3, x4]))
    end
end
