using Stheno: derivative

@timedtestset "derivative" begin
    x1 = randn(4)
    x2 = randn(5)
    f = GP(0.5 * stretch(EQ(), 0.1), GPC())
    df = derivative(f)

    # Check that the derivative of a zero-mean GP is in fact zero.
    @test all(mean(df(x1)) .== 0)

    # Check that the kernel of the derivative process agrees with a numerical approx.
    for n1 in eachindex(x1), n2 in eachindex(x2)

        # Verify (approximate) correctness of covariance of df.
        cov_df2_dx1_dx2_fd = FiniteDifferences.fdm(
            central_fdm(5, 1),
            _x1 -> begin
                FiniteDifferences.fdm(
                    central_fdm(5, 1),
                    _x2 -> first(cov(f([_x1]), f([_x2]))),
                    x2[n2],
                )
            end,
            x1[n1],
        )
        cov_df2_dx1_dx2_kernel = first(cov(df([x1[n1]]), df([x2[n2]])))
        @test cov_df2_dx1_dx2_fd ≈ cov_df2_dx1_dx2_kernel atol=1e6 rtol=1e-6

        # Verify correctness of covariance between f and df.
        cov_f_df_x1_x2_fd = FiniteDifferences.fdm(
            central_fdm(5, 1),
            _x2 -> first(cov(f([x1[n1]]), f([_x2]))),
            x2[n2],
        )
        cov_f_df_x1_x2_kernel = first(cov(f([x1[n1]]), df([x2[n2]])))
        @test isapprox(cov_f_df_x1_x2_fd, cov_f_df_x1_x2_kernel; atol=1e-6, rtol=1e-6)
    end

    @timedtestset "Consistency Tests" begin
        rng, P, Q = MersenneTwister(123456), 3, 5
        x0 = collect(range(-1.0, 1.0; length=P))
        x1 = collect(range(-0.5, 1.5; length=Q))
        x2 = randn(rng, Q)
        x3 = randn(rng, P)

        abstractgp_interface_tests(df, f, x0, x1, x2, x3)
    end
end
