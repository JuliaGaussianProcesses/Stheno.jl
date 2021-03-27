@timedtestset "gaussian_process_probabilistic_programme" begin

    # Build a toy collection of processes.
    gpc = GPC()
    f1 = wrap(GP(sin, SEKernel()), gpc)
    f2 = wrap(GP(cos, Matern52Kernel()), gpc)
    f3 = f1 + 3 * f2

    # Use them to build a programme.
    f = GPPP(Dict(f1.n => f1, f2.n => f2, f3.n => f3), gpc)

    # Check for consistency between the two for some pairs of inputs.
    x0 = GPPPInput(f1.n, randn(4))
    x1 = GPPPInput(f3.n, randn(3))

    # The same answers should be obtained manually or via the GPPP.
    @testset "External Consistency" begin
        @test mean(f1, x0.x) == mean(f, x0)
        @test mean(f3, x1.x) == mean(f, x1)

        @test cov(f1, x0.x) == cov(f, x0)
        @test cov(f3, x1.x) == cov(f, x1)

        @test cov(f1, f3, x0.x, x1.x) == cov(f, x0, x1)
        @test cov_diag(f3, f1, x1.x, x0.x) == cov_diag(f, x1, x0)
    end

    # The GPPP must be self-consistent like any other AbstractGP.
    @timedtestset "Internal Conistency" begin

        atol=1e-9
        rtol=1e-9

        m = mean(f, x0)
        @test m isa AbstractVector{<:Real}
        @test length(m) == length(x0)

        @assert length(x0) ≠ length(x1)

        # Check that unary cov is consistent with binary cov and conforms to the API
        K_x0 = cov(f, x0)
        @test K_x0 isa AbstractMatrix{<:Real}
        @test size(K_x0) == (length(x0), length(x0))
        @test K_x0 ≈ cov(f, x0, x0) atol=atol rtol=rtol
        @test minimum(eigvals(K_x0)) > -atol
        @test K_x0 ≈ K_x0' atol=atol rtol=rtol

        # Check that single-process binary cov is consistent with single-process binary-cov
        K_x0_x1 = cov(f, x0, x1)
        @test K_x0_x1 isa AbstractMatrix{<:Real}
        @test size(K_x0_x1) == (length(x0), length(x1))

        # Check that single-process binary cov_diag is consistent.
        K_x0_x0_diag = cov_diag(f, x0, x0)
        @test K_x0_x0_diag isa AbstractVector{<:Real}
        @test length(K_x0_x0_diag) == length(x0)
        @test K_x0_x0_diag ≈ diag(cov(f, x0, x0)) atol=atol rtol=rtol

        # Check that unary cov_diag conforms to the API and is consistent with unary cov
        K_x0_diag = cov_diag(f, x0)
        @test K_x0_diag isa AbstractVector{<:Real}
        @test length(K_x0_diag) == length(x0)
        @test K_x0_diag ≈ diag(cov(f, x0)) atol=atol rtol=rtol
    end

    @testset "gppp macro" begin

        # Declare a GPPP using the helper functionality.
        f = @gppp let
            f1 = GP(SEKernel())
            f2 = GP(Matern52Kernel())
            f3 = f1 + f2
        end

        x0 = GPPPInput(:f1, randn(5))
        x1 = GPPPInput(:f3, randn(4))
        cov(f(x0), f(x1))

        y = rand(f(x0, 0.1))
        @show logpdf(f(x0, 0.1), y)
    end
end
