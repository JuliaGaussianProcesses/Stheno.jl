import Distributions: logpdf, rand

const FiniteGP = GP{<:FiniteMean, <:FiniteKernel}
const _finite_gp_rng = MersenneTwister(123456);

# f isa FiniteGP => we don't need input locations. Necessary for compatability with Turing.
logpdf(f::FiniteGP, y::AbstractVector{<:Real}) = logpdf(f.args[1], f.args[2], y)
rand(f::FiniteGP) = rand(_finite_gp_rng, f.args[1], f.args[2])
