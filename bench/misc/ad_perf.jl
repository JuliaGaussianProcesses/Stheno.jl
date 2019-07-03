using Stheno, Zygote, BenchmarkTools
using Stheno: @model, BlockData, cross

const x = collect(range(-5.0, 5.0; length=50));
const x1 = randn(25)
const x2 = randn(25)

θ = (m=5.0, l=1.3);

@model foo(θ) = GP(ConstMean(θ.m), eq(l=θ.l));



#
# Test single-process versions of operations on foo.
#

y = rand(foo(θ)(x, 0.1));

@code_warntype mean(foo(θ)(x, 0.1))
@benchmark mean(foo($θ)($x, 0.1))

@code_warntype cov(foo(θ)(x, 0.1))
@benchmark cov(foo($θ)($x, 0.1))

@code_warntype rand(foo(θ)(x, 0.1))
@benchmark rand(foo($θ)($x, 0.1))

@code_warntype logpdf(foo(θ)(x, 0.1), y)
@benchmark logpdf(foo($θ)($x, 0.1), $y)



#
# Test multi-process versions of operations on foo.
#

function test_cross(θ)
    f = foo(θ)
    return cross([f, f])
end

@code_warntype test_cross(θ)

test_kernels(f) = kernel.([f, f], permutedims([f, f]))
@code_warntype test_kernels(f)

test_kernel(f) = Stheno.kernel(f, f)
@code_warntype test_kernel(f)

function test_multi_mean(θ, x1, x2)
    f = foo(θ)
    return mean(cross([f, f])(BlockData([x1, x2]), 0.1))
end

@code_warntype test_multi_mean(θ, x1, x2)

function test_multi_rand(θ, x1, x2)
    f = foo(θ)
    fx1 = f(x1, 0.1)
    fx2 = f(x2, 0.1)
    return rand([fx1, fx2])
end

y1, y2 = test_multi_rand(θ, x1, x2);

@code_warntype test_multi_rand(θ, x, x)
@benchmark test_multi_rand($θ, $x1, $x2)

function test_multi_logpdf(θ, x1, x2, y1, y2)
    f = foo(θ)
    fx1 = f(x1, 0.1)
    fx2 = f(x2, 0.1)
    return logpdf([fx1 ← y1, fx2 ← y2])
end

@code_warntype test_multi_logpdf(θ, x1, x2, y1, y2)
@benchmark test_multi_logpdf($θ, $x1, $x2, $y1, $y2)

