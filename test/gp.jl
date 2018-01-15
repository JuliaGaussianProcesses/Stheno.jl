@testset "gp" begin

    # Test the creation of indepenent GPs.
    import Stheno.Constant
    let rng = MersenneTwister(123456)

        # Specification for three independent GPs.
        μ1, μ2, μ3 = CustomMean.((sin, cos, tan))
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
        f1, f2, f3 = GP.([μ1, μ2, μ3], [k1, k2, k3], GPC())

        @test mean(f1) == μ1
        @test mean(f2) == μ2
        @test mean(f3) == μ3

        @test kernel(f1) == k1
        @test kernel(f2) == k2
        @test kernel(f3) == k3

        @test kernel(f1, f1) == k1
        @test kernel(f1, f2) == Constant(0.0)
        @test kernel(f1, f3) == Constant(0.0)
        @test kernel(f2, f1) == Constant(0.0)
        @test kernel(f2, f2) == k2
        @test kernel(f2, f3) == Constant(0.0)
        @test kernel(f3, f1) == Constant(0.0)
        @test kernel(f3, f2) == Constant(0.0)
        @test kernel(f3, f3) == k3
    end

    # Test a generic toy problem.
    let
        rng = MersenneTwister(123456)
        N, S = 5, 100000
        μ_vec, x = randn(rng, N), randn(rng, N)
        μ, k = FiniteMean(CustomMean(identity), μ_vec), Finite(EQ(), x)
        d = GP(μ, k, GPC())

        @test mean(d) == μ
        @test mean(d).(1:N) == μ_vec
        @test dims(d) == N
        @test kernel(d).(1:N, (1:N)') == EQ().(x, x')

        x̂ = sample(rng, d, S)
        @test size(x̂) == (N, S)
        @test maximum(abs.(mean(x̂, 2) - mean(d).(1:N))) < 1e-2
        Σ = broadcast(kernel(d), collect(1:N), Transpose(collect(1:N)))
        @test maximum(abs.(cov(x̂, 2) - Σ)) < 1e-2
    end
end
