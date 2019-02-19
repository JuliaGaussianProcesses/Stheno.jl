using Stheno: GPC, PseudoPointsCov, project, Titsias, optimal_q, pw, Xt_invA_X, Xt_invA_Y

# Test Titsias implementation by checking that it (approximately) recovers exact inference
# when M = N and Z = X.
@testset "Titsias" begin

    @testset "optimal_q" begin

        rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 10, 1001, 15, 1e-1, GPC()
        x = collect(range(-3.0, 3.0, length=N))
        f = GP(sin, eq(), gpc)
        y = rand(rng, f(x, σ²))

        # Compute approximate posterior suff. stats.
        m′u, Λ, U = optimal_q(f(x, σ²)←y, f(x))
        f′ = f | (f(x, σ²) ← y)

        # Check that exact and approx. posteriors are close in this case.
        @test m′u ≈ mean(f′(x))
        @test U ≈ cholesky(cov(f(x))).U
        @test Λ.U ≈ cholesky((U' \ cov(f(x))) * (U' \ cov(f(x)))' ./ σ² + I).U
    end

    @testset "project" begin
        rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 2, 3, 4, 1e-1, GPC()
        x = collect(range(-3.0, 3.0, length=N))
        x′ = collect(range(-3.0, 3.0, length=N′))
        z = collect(range(-3.0, 3.0, length=Nz))
        f = GP(sin, eq(), gpc)
        y = rand(rng, f(x, σ²))

        m′u, Λ, U = optimal_q(f(x, σ²)←y, f(z))
        u = GP(PseudoPointsCov(Λ, U), gpc)
        kg, kh = eq(l=0.5), eq(l=1.1)
        g, h = project(kg, u, z), project(kh, u, z)

        @test iszero(mean(g))
        @test iszero(mean(h))

        @test pw(kernel(g), x) ≈ Xt_invA_Y(U' \ pw(kg, z, x), Λ, U' \ pw(kg, z, x))
        @test pw(kernel(g), x′) ≈ Xt_invA_Y(U' \ pw(kg, z, x′), Λ, U' \ pw(kg, z, x′))

        @test pw(kernel(g), x, x′) ≈ Xt_invA_Y(U' \ pw(kg, z, x), Λ, U' \ pw(kg, z, x′))
        @test pw(kernel(h), x, x′) ≈ Xt_invA_Y(U' \ pw(kh, z, x), Λ, U' \ pw(kh, z, x′))

        @test pw(kernel(g, h), x, x′) ≈ Xt_invA_Y(U' \ pw(kg, z, x), Λ, U' \ pw(kh, z, x′))
        @test pw(kernel(h, g), x, x′) ≈ Xt_invA_Y(U' \ pw(kh, z, x), Λ, U' \ pw(kg, z, x′))
    end

    @testset "single conditioning" begin
        rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 11, 10, 11, 1e-1, GPC()
        x = collect(range(-3.0, 3.0, length=N))
        x′ = collect(range(-3.0, 3.0, length=N′))
        z = x

        # Generate toy problem.
        f = GP(sin, eq(), gpc)
        y = rand(f(x, σ²))

        # Exact conditioning.
        f′ = f | (f(x, σ²)←y)

        # Approximate conditioning that should yield almost exact results.
        titsias = Titsias(f(x, σ²)←y, f(z))
        f′_approx = f | titsias
        g′_approx = f | titsias

        @test mean(f′(x′)) ≈ mean(f′_approx(x′))
        @test cov(f′(x′)) ≈ cov(f′_approx(x′))
        @test cov(f′(x′), f′(x)) ≈ cov(f′_approx(x′), f′_approx(x))
        @test cov(f′(x′), f′(x)) ≈ cov(f′_approx(x′), g′_approx(x))
        @test cov(f′(x′), f′(x)) ≈ cov(g′_approx(x′), f′_approx(x))
    end

    # @testset "multiple conditioning" begin
    #     rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 11, 10, 11, 1e-1, GPC()
    #     x = collect(range(-3.0, 3.0, length=N))
    #     x′ = collect(range(-3.0, 3.0, length=N′))
    #     x̂ = collect(range(-4.0, 4.0, length=N))
    #     z = copy(x)

    #     # Construct toy problem.
    #     f = GP(sin, eq(), gpc)
    #     y, y′ = rand(rng, [f(x, σ²), f(x′, σ²)])

    #     # Perform approximate inference with concatenated inputs.
    #     f′_concat = f | Titsias(f(vcat(x, x′), σ²)←vcat(y, y′), f(z))

    #     # Perform approximate inference with multiple observations.
    #     f′_multi = f | Titsias(f(x, σ²)←y, f(x′, σ²)←y′, f(z))

    #     # Check that the above agree.
    #     @test mean(f′_concat(x̂)) == mean(f′_concat(x̂))
    #     @test cov(f′_concat(x̂)) == cov(f′_concat(x̂))
    # end

    # rng, N, N′, D, σ² = MersenneTwister(123456), 2, 3, 5, 1e-1
    # X_, X′_ = randn(rng, D, N), randn(rng, D, N′)
    # X, X′, Z = ColsAreObs(X_), ColsAreObs(X′_), ColsAreObs(randn(rng, D, N + N′))
    # μ, k, XX′ = ConstMean(1.0), eq(), ColsAreObs(hcat(X_, X′_))

    # # Construct toy problem.
    # gpc = GPC()
    # f = GP(μ, k, gpc)
    # y = f + GP(noise(α=sqrt(σ²)), gpc)
    # ŷ = rand(rng, y(XX′))

    # # Compute exact posterior.
    # f′XX′ = f(XX′) | (y(XX′)←ŷ)


    # # Compute conditioner and exact posterior compute at test points.
    # conditioner = Stheno.Titsias(f(XX′), μᵤ, Σᵤᵤ)
    # f′Z = f(Z) | (y(XX′)←ŷ)
    # f′Z_approx = f(Z) | conditioner

    # # Check that exact and approximate posteriors match up.
    # @test isapprox(mean(f′Z), mean(f′Z_approx); rtol=1e-4)
    # @test isapprox(cov(f′Z), cov(f′Z_approx); rtol=1e-4)


    # # Check that Titsias with BlockGP works the same as Titsias with regular GP.
    # ŷX, ŷX′ = ŷ[1:N], ŷ[N+1:end]

    # fb, ŷb = BlockGP([f(X), f(X′)]), BlockVector([ŷX, ŷX′])
    # μb, Σb = Stheno.optimal_q(fb, ŷb, fb, sqrt(σ²))

    # @test μb isa BlockVector
    # @test Stheno.unbox(Σb) isa Symmetric
    # @test Stheno.unbox(Stheno.unbox(Σb)) isa AbstractBlockMatrix
    # @test μb ≈ μᵤ
    # @test Σb ≈ Σᵤᵤ

    # # Test that conditioning is indifferent to choice of Blocks.
    # conditioner_blocked = Stheno.Titsias(fb, μb, Σb)
    # f′Zb = f(BlockData([Z])) | conditioner_blocked

    # @test isapprox(mean(f′Z), mean(f′Zb); rtol=1e-4)
    # @test isapprox(cov(f′Z), cov(f′Zb); rtol=1e-4)
end
