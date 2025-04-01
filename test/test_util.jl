"""
    abstractgp_interface_tests(
        f::AbstractGP,
        f′::AbstractGP,
        x0::AbstractVector,
        x1::AbstractVector,
        x2::AbstractVector,
        x3::AbstractVector;
        atol=1e-9, rtol=1e-9,
    )

Check that the `AbstractGP` interface is at least implemented for `f` and is
self-consistent. `x0` and `x1` must be valid inputs for `f`. `x2` and `x3` must be a valid
input for `f′`.
"""
function abstractgp_interface_tests(
    f::AbstractGP,
    f′::AbstractGP,
    x0::AbstractVector,
    x1::AbstractVector,
    x2::AbstractVector,
    x3::AbstractVector;
    atol=1e-9,
    rtol=1e-9,
)
    m = mean(f, x0)
    @test m isa AbstractVector{<:Real}
    @test length(m) == length(x0)

    @assert length(x0) ≠ length(x1)
    @assert length(x0) ≠ length(x2)
    @assert length(x0) == length(x3)

    # Check that binary cov conforms to the API
    K_ff′_x0_x2 = cov(f, f′, x0, x2)
    @test K_ff′_x0_x2 isa AbstractMatrix{<:Real}
    @test size(K_ff′_x0_x2) == (length(x0), length(x2))
    @test K_ff′_x0_x2 ≈ cov(f′, f, x2, x0)'

    # Check that unary cov is consistent with binary cov and conforms to the API
    K_x0 = cov(f, x0)
    @test K_x0 isa AbstractMatrix{<:Real}
    @test size(K_x0) == (length(x0), length(x0))
    @test K_x0 ≈ cov(f, f, x0, x0) atol = atol rtol = rtol
    @test minimum(eigvals(K_x0)) > -atol
    @test K_x0 ≈ K_x0' atol = atol rtol = rtol

    # Check that single-process binary cov is consistent with binary-process binary-cov
    K_x0_x1 = cov(f, x0, x1)
    @test K_x0_x1 isa AbstractMatrix{<:Real}
    @test size(K_x0_x1) == (length(x0), length(x1))
    @test K_x0_x1 ≈ cov(f, f, x0, x1)

    # Check that binary var conforms to the API and is consistent with binary cov
    K_x0_x3_diag = var(f, f′, x0, x3)
    @test K_x0_x3_diag isa AbstractVector{<:Real}
    @test length(K_x0_x3_diag) == length(x0)
    @test K_x0_x3_diag ≈ diag(cov(f, f′, x0, x3)) atol = atol rtol = rtol
    @test K_x0_x3_diag ≈ var(f′, f, x3, x0) atol = atol rtol = rtol

    # Check that unary-binary var is consistent.
    K_x0_x0_diag = var(f, x0, x0)
    @test K_x0_x0_diag isa AbstractVector{<:Real}
    @test length(K_x0_x0_diag) == length(x0)
    @test K_x0_x0_diag ≈ diag(cov(f, x0, x0)) atol = atol rtol = rtol

    # Check that unary var conforms to the API and is consistent with unary cov
    K_x0_diag = var(f, x0)
    @test K_x0_diag isa AbstractVector{<:Real}
    @test length(K_x0_diag) == length(x0)
    @test K_x0_diag ≈ diag(cov(f, x0)) atol = atol rtol = rtol
end
