using RecipesBase

@recipe f(gp::Stheno.AbstractGP, x::Array) = gp(x)
@recipe f(gp::Stheno.AbstractGP, x::AbstractRange) = gp(x)
@recipe f(gp::Stheno.AbstractGP, xmin::Number, xmax::Number) = (gp, range(xmin, xmax, length=1000))
@recipe function f(gp::Stheno.FiniteGP; samples=0, sample_seriestype=:line)
    x = gp.x
    f = gp.f
    ms = marginals(gp)

    @series begin
        μ = mean.(ms)
        σ = std.(ms)
        ribbon := 3σ
        fillalpha --> 0.3
        width --> 2
        x, μ
    end

    if samples>0
        @series begin
            samples = rand(f(x, 1e-9), samples)

            seriestype --> sample_seriestype

            linealpha := 0.2

            markershape --> :circle
            markerstrokewidth --> 0.0
            markersize --> 0.5
            markeralpha --> 0.3

            label := ""

            x, samples
        end
    end
end
