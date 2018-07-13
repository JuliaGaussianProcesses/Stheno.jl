using Stheno
using Stheno.@model
using ToeplitzMatrices: SymmetricToeplitz

@testset "Toeplitz Integration" begin

xl = linspace(-5.0, 5.0, 10_000);
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
@test cov(f(xl)) isa LazyPDMat{<:Real, <:SymmetricToeplitz}
@test cov(y(xl)) isa LazyPDMat{<:Real, <:SymmetricToeplitz}

# @benchmark rand($f($xl))
# @benchmark rand($f($x))

# @benchmark logpdf($f($xl), $y)
# @benchmark logpdf($f($x), $y)





end
