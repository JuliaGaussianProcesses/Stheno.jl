using Nabla, Stheno, Random

rng, N, D = MersenneTwister(123456), 3, 2
X = randn(rng, N, D)

bar(X) = sum(X * X')
bar(X)
∇(bar)(X)


function foo(c)
    return sum(mean(ConstantMean(c), X))
end

foo(5.3)
∇(foo)(5.3)

