using Stheno, Plots
plotly();



############################## Define model and data ################################

# Define additive model.
σ_noise, gpc = 1e-1, GPC();
f₁ = GP(pick_dims(EQ(), 1), gpc);
f₂ = GP(pick_dims(EQ(), 2), gpc);
f₃ = f₁ + f₂;
y = f₃ + GP(Noise(σ_noise^2), gpc);

# Sample from marginal process to generate toy data.
rng, S = MersenneTwister(123456), 25;
X = ColsAreObs(randn(2, 100));
ŷ = rand(rng, y(X));



############################### Set up plotting ##################################

# Define plotting locations.
P, Q, pos, b = 50, 50, 1, 4.0;
Ps, Qs, Xp = linspace(-b, b, P), linspace(-b, b, Q), Matrix{Float64}(2, P * Q);
for p in 1:P, q in 1:Q
    Xp[1, pos] = Ps[p]
    Xp[2, pos] = Qs[q]
    pos += 1
end
Xp = ColsAreObs(Xp);

exact_plot, approx_plot = plot(), plot();



################################ Do exact inference ################################

@show logpdf(y(X), ŷ);
f₁′, f₂′, f₃′ = (f₁, f₂, f₃) | (y(X) ← ŷ);
f₁′Xp, f₂′Xp, f₃′Xp = rand(rng, [f₁′(Xp), f₂′(Xp), f₃′(Xp)], S);
μ₁′, σ₁′ = marginals(f₁′(Xp));
μ₂′, σ₂′ = marginals(f₂′(Xp));
μ₃′, σ₃′ = marginals(f₃′(Xp));



################################ Do approximate inference ###############################

M = 15;
Z₁ = ColsAreObs(hcat(linspace(-b, b, M), zeros(M))');
Z₂ = ColsAreObs(hcat(zeros(M), linspace(-b, b, M))');
u₁, u₂ = f₁(Z₁), f₂(Z₂);
u = BlockGP([u₁, u₂]);
@show elbo(f₃(X), ŷ, u, σ_noise);
μ′u, Σ′u = Stheno.optimal_q(f₃(X), ŷ, u, σ_noise);
conditioner = Stheno.Titsias(u, μ′u, Σ′u, gpc);
f₁′u, f₂′u, f₃′u = (f₁, f₂, f₃) | conditioner;


