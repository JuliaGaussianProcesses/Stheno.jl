using Stheno: GPC, PPC, project, Titsias, optimal_q, pw, Xt_invA_X, Xt_invA_Y

# Test Titsias implementation by checking that it (approximately) recovers exact inference
# when M = N and Z = X.
@testset "Titsias" begin

    @testset "optimal_q (single conditioning)" begin

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

    @testset "optimal_q (multiple conditioning)" begin

        # rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 10, 1001, 15, 1e-1, GPC()
        rng, N, N′, σ², gpc = MersenneTwister(123456), 5, 7, 1e-1, GPC()
        xx′ = collect(range(-3.0, stop=3.0, length=N+N′))
        idx = randperm(rng, length(xx′))[1:N]
        idx_1, idx_2 = idx, setdiff(1:length(xx′), idx)
        x, x′ = xx′[idx_1], xx′[idx_2]

        f = GP(sin, eq(), gpc)
        y, y′ = rand(rng, [f(x, σ²), f(x′, σ²)])

        # Compute approximate posterior suff. stats.
        m′u, Λ, U = optimal_q((f(x, σ²)←y, f(x′, σ²)←y′), f(xx′))
        f′ = f | (f(x, σ²) ← y, f(x′, σ²)←y′)

        # Check that exact and approx. posteriors are close in this case.
        @test m′u ≈ mean(f′(xx′))
        @test U ≈ cholesky(cov(f(xx′))).U
        @test Λ.U ≈ cholesky((U' \ cov(f(xx′))) * (U' \ cov(f(xx′)))' ./ σ² + I).U
    end

    @testset "project" begin
        rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 2, 3, 4, 1e-1, GPC()
        x = collect(range(-3.0, 3.0, length=N))
        x′ = collect(range(-3.0, 3.0, length=N′))
        z = collect(range(-3.0, 3.0, length=Nz))
        f = GP(sin, eq(), gpc)
        y = rand(rng, f(x, σ²))

        m′u, Λ, U = optimal_q(f(x, σ²)←y, f(z))
        u = GP(PPC(Λ, U), gpc)
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
        rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 2, 3, 2, 1e-1, GPC()
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

    @testset "multiple conditioning" begin
        rng, N, N′, Nz, σ², gpc = MersenneTwister(123456), 11, 10, 11, 1e-1, GPC()
        x = collect(range(-3.0, 3.0, length=N))
        x′ = collect(range(-3.0, 3.0, length=N′))
        x̂ = collect(range(-4.0, 4.0, length=N))
        z = copy(x)

        # Construct toy problem.
        f = GP(sin, eq(), gpc)
        y, y′ = rand(rng, [f(x, σ²), f(x′, σ²)])

        # Perform approximate inference with concatenated inputs.
        f′_concat = f | Titsias(f(vcat(x, x′), σ²)←vcat(y, y′), f(z))

        # Perform approximate inference with multiple observations.
        f′_multi = f | Titsias((f(x, σ²)←y, f(x′, σ²)←y′), f(z))

        # Check that the above agree.
        @test mean(f′_concat(x̂)) == mean(f′_concat(x̂))
        @test cov(f′_concat(x̂)) == cov(f′_concat(x̂))
    end
end
