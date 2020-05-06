"""
when Flux is used, include this file will automatically add functor to existing
kernels in Stheno.
"""

using .Flux
using .Flux: @functor

@functor GP
@functor Parameter

kernels = [:EQ, :PerEQ, :Exp, :Matern12, :Matern32, :Matern52, :RQ, :Cosine, :Linear, :Poly, :GammaExp, :Wiener,
    :WienerVelocity, :Sum, :Product, :Scaled, :Stretched]

for k in kernels
    @eval @functor $k
end

