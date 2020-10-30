using Stheno: differentiate

@testset "derivative" begin

    # Construct derivative GP.
    f = GP(EQ(), GPC())
    ∂f = differentiate(f, ForwardDiff.derivative)

    # Specify inputs and verify internal consistency.
    x0 = collect(range(-3.0, 3.0; length=10))
    x1 = collect(range(-3.1, 2.2; length=9))
    x2 = collect(range(-3.2, 2.3; length=9))
    x3 = collect(range(-3.5, 3.4; length=10))
    abstractgp_interface_tests(∂f, f, x0, x1, x2, x3)

    # Verify that a constrained posterior gives the correct result.
    xtr = range(0.0, π; length=100)
    ytr = sin.(xtr)

    ∂f′ = ∂f | (f(xtr, 1e-12) ← ytr)
    xte = rand(10) .* π
    ∂f′_xte = rand(∂f′(xte))
    @test all(isapprox.(∂f′_xte, cos.(xte); rtol=1e-5, atol=1e-5))
end
