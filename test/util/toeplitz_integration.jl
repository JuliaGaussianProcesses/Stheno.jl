using Stheno
using Stheno: @model, LazyPDMat
using ToeplitzMatrices: SymmetricToeplitz

@testset "Toeplitz Integration" begin

xl = range(-5.0; stop=5.0, length=1000);
x = collect(xl);

# Simplest possible model.
@model function simple()
    return GP(EQ())
end

# Slightly less simple model.
@model function noisy_regression()
    f = GP(EQ())
    y = f + GP(Noise(0.1))
    return f, y
end

f = simple();
y = rand(f(xl));
@test cov(f(xl)) isa LazyPDMat{<:Real, <:SymmetricToeplitz}

f, y = noisy_regression()
ŷ = rand(y(xl))
@test cov(f(xl)) isa LazyPDMat{<:Real, <:SymmetricToeplitz}
@test cov(y(xl)) isa LazyPDMat{<:Real, <:SymmetricToeplitz}

@test cov(f(xl)) ≈ cov(f(x))
@test cov(y(xl)) ≈ cov(y(x))

fb_xl = BlockGP([f(xl), y(xl)])
fb_x = BlockGP([f(x), y(x)])






end
