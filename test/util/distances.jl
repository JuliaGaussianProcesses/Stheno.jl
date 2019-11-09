using Distances: Euclidean, SqEuclidean

function dist_tests(
    Dist,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    Ȳsq::AbstractMatrix{<:Real},
    x1::AbstractVector,
    x2::AbstractVector,
    x3::AbstractVector;
    rtol=_rtol,
    atol=_atol,
)
    # Check inputs have valid lengths.
    @assert length(x1) == length(x2)
    @assert length(x1) != length(x3)

    @assert length(ȳ) == length(x1)
    @assert size(Ȳ, 1) == length(x1)
    @assert size(Ȳ, 2) == length(x3)
    @assert size(Ȳsq, 1) == size(Ȳsq, 2)
    @assert size(Ȳsq, 1) == length(x1)

    # Binary == Unary
    @test ew(Dist, x1, x1) ≈ ew(Dist, x1)
    @test pw(Dist, x1, x1) ≈ pw(Dist, x1)

    # Symmetry
    @test ew(Dist, x1, x2) ≈ ew(Dist, x2, x1)
    @test pw(Dist, x1, x3) ≈ collect(pw(Dist, x3, x1)')

    # Diagonal of pw is ew
    @test ew(Dist, x1, x2) ≈ diag(pw(Dist, x1, x2))
    @test ew(Dist, x1) ≈ diag(pw(Dist, x1))

    # adjoints
    adjoint_test((x, x′)->ew(Dist, x, x′), ȳ, x1, x2; rtol=rtol, atol=atol)
    adjoint_test((x, x′)->pw(Dist, x, x′), Ȳ, x1, x3; rtol=rtol, atol=atol)

    adjoint_test(x->ew(Dist, x), ȳ, x1; rtol=rtol, atol=atol)
    adjoint_test(x->pw(Dist, x), Ȳsq, x1; rtol=rtol, atol=atol)
end

@testset "distances" begin
    @testset "$Dist" for Dist in [Euclidean(), SqEuclidean()]
        @testset "Vector" begin
            rng = MersenneTwister(123456)
            P, Q = 5, 3
            dist_tests(
                Dist,
                randn(rng, P),
                randn(rng, P, Q),
                randn(rng, P, P),
                collect(range(-2.0, 2.0; length=P)),
                randn(rng, P),
                randn(rng, Q),
            )
        end
        @testset "ColVecs" begin
            rng = MersenneTwister(123456)
            P, Q, D = 5, 3, 2
            dist_tests(
                Dist,
                randn(rng, P),
                randn(rng, P, Q),
                randn(rng, P, P),
                ColVecs(randn(rng, D, P)),
                ColVecs(randn(rng, D, P)),
                ColVecs(randn(rng, D, Q)),
            )
        end
    end
end
