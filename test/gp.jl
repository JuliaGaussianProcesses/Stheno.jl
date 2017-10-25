@testset "gp" begin

    # Test the creation of indepenent GPs.
    let rng = MersenneTwister(123456)

        # Specification for three independent GPs.
        μ1, μ2, μ3 = sin, cos, tan
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
        μ, k = n::Int->μ_vec[n], (m::Int, n::Int)->EQ()(x[m], x[n])
        d = Normal(μ, k, N, GPC())

        @test mean(d) == μ
        @test mean(d).(1:N) == μ_vec
        @test dims(d) == N
        for m in 1:N, n in 1:N
            @test kernel(d)(m, n) == EQ()(x[m], x[n])
        end

        x̂ = sample(rng, d, S)
        @test size(x̂) == (N, S)
        @test maximum(abs.(mean(x̂, 2) - mean(d).(1:N))) < 1e-2
        Σ = broadcast(kernel(d), collect(1:N), RowVector(collect(1:N)))
        @test maximum(abs.(cov(x̂, 2) - Σ)) < 1e-2
    end

    # Test mean_vector.
    let rng = MersenneTwister(123456)
        x, x′ = randn(rng, 5), randn(rng, 10)
        f = GP(sin, EQ(), GPC())
        @test mean_vector(f(x)) == sin.(x)
        @test mean_vector([f(x), f(x′)]) == vcat(sin.(x), sin.(x′))
    end

    # Test deterministic features of `sample`.
    let rng = MersenneTwister(123456)
        x, x′ = randn(rng, 5), randn(rng, 10)
        f = GP(sin, EQ(), GPC())
        @test length(sample(rng, f(x))) == length(x)
        @test size(sample(rng, f(x), 10)) == (length(x), 10)

        @test length(sample(rng, [f(x), f(x′)])[1]) == length(x)
        @test length(sample(rng, [f(x), f(x′)])[2]) == length(x′)
        @test size(sample(rng, [f(x), f(x′)], 10)[1]) == (length(x), 10)
        @test size(sample(rng, [f(x), f(x′)], 10)[2]) == (length(x′), 10)
    end
end
