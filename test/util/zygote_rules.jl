using FDM, Zygote, Distances, Random, LinearAlgebra, FillArrays
using Stheno: chol

@testset "zygote_rules" begin

# Check FillArrays work as expected.
let
    @test Zygote.gradient(x->sum(Fill(x, 10)), randn())[1] == 10
    @test Zygote.gradient(x->sum(Fill(x, (10, 3, 4))), randn())[1] == 10 * 3 * 4

    # Test unary broadcasting gradients.
    x = randn()
    out, back = Zygote.forward(x->exp.(x), Fill(x, 10))
    @test out isa Fill
    @test out == Fill(exp(x), 10)
    @test back(Ones(10))[1] isa Fill
    @test back(Ones(10))[1] == Ones(10) .* exp(x)
    @test back(ones(10))[1] isa Vector
    @test back(ones(10))[1] == ones(10) .* exp(x)
end

# # Check squared-euclidean distance implementation (AbstractMatrix)
# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

#     # Check sqeuclidean.
#     x, y = randn(rng, D), randn(rng, D)
#     f_el_1 = x->sqeuclidean(x, y)
#     @test all(abs.(Zygote.gradient(f_el_1, x)[1] .- FDM.grad(fdm, f_el_1, x)) .< 1e-8) 

#     f_el_2 = y->sqeuclidean(x, y)
#     @test all(abs.(Zygote.gradient(f_el_2, y)[1] .- FDM.grad(fdm, f_el_2, y)) .< 1e-8)

#     # Check binary colwise
#     X, Y = randn(rng, D, P), randn(rng, D, P)
#     f_col_1 = X->sum(colwise(SqEuclidean(), X, Y))
#     @test all(abs.(Zygote.gradient(f_col_1, X)[1] .- FDM.grad(fdm, f_col_1, X)) .< 1e-8)

#     f_col_2 = Y->sum(colwise(SqEuclidean(), X, Y))
#     @test all(abs.(Zygote.gradient(f_col_2, Y)[1] .- FDM.grad(fdm, f_col_2, Y)) .< 1e-8)
# end

# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

#     # Generate differing-length vectors for pairwise.
#     X, Y = randn(rng, D, P), randn(rng, D, Q)

#     # Check first argument of binary pairwise.
#     f_pw_1 = X->sum(pairwise(SqEuclidean(), X, Y))
#     @test all(Zygote.gradient(f_pw_1, X)[1] .- FDM.grad(fdm, f_pw_1, X) .< 1e-8)

#     # Check second argument of binary pairwise.
#     f_pw_2 = Y->sum(pairwise(SqEuclidean(), X, Y))
#     @test all(Zygote.gradient(f_pw_2, Y)[1] .- FDM.grad(fdm, f_pw_2, Y) .< 1e-8)

#     # Check unary pairwise.
#     @test Zygote.gradient(X->sum(pairwise(SqEuclidean(), X)), X)[1] ≈
#         Zygote.gradient(X->sum(pairwise(SqEuclidean(), X, X)), X)[1]
# end

# # Check squared-euclidean distance implementation (AbstractVector)
# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q = MersenneTwister(123456), 10, 9
#     x, y = randn(rng, P), randn(rng, Q)

#     # Check first argument of binary pairwise.
#     f = x->sum(pairwise(SqEuclidean(), x, y))
#     @test all(Zygote.gradient(f, x)[1] .- FDM.grad(fdm, f, x) .< 1e-8)

#     # Check second argument of binary pairwise.
#     f = y->sum(pairwise(SqEuclidean(), x, y))
#     @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

#     # Check unary pairwise.
#     @test Zygote.gradient(x->sum(pairwise(SqEuclidean(), x)), x)[1] ≈
#         Zygote.gradient(x->sum(pairwise(SqEuclidean(), x, x)), x)[1]
# end

# # Check that \ and / work.
# let
#     fdm = central_fdm(5, 1)
#     rng, P, Q = MersenneTwister(123456), 10, 9
#     X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)

#     # \

#     # Check first argument sensitivity in matrix case.
#     f = X->sum(X \ Y)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in matrix case.
#     f = Y->sum(X \ Y)
#     @test all(Zygote.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

#     # Check first argument sensitivity in vector case.
#     f = X->sum(X \ y)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in vector case.
#     f = y->sum(X \ y)
#     @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

#     # /

#     # Check first argumeant sensitivity in matrix case.
#     f = X->sum(Y' / X)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in matrix case.
#     f = Y->sum(Y' / X)
#     @test all(Zygote.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

#     # Check first argument sensitivity in vector case.
#     f = X->sum(y' / X)
#     @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

#     # Check second argument sensitivity in vector case.
#     f = y->sum(y' / X)
#     @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)
# end

# # Check that Symmetric works as expected.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 7
#     A = randn(rng, P, P)

#     f = A->sum(Symmetric(A))
#     @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
# end

# # Check that cholesky works as expected.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 7
#     A = randn(rng, P, P)
#     Σ = A'A + 1e-6I

#     f = Σ->sum(chol(Symmetric(Σ)))
#     @test all(Zygote.gradient(f, Σ)[1] .- FDM.grad(fdm, f, Σ) .< 1e-8)
# end

# # Check that `diag` behaves sensibly.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 10
#     A = randn(rng, P, P)

#     f = A->sum(diag(A))
#     @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
# end

# # Check that `logdet` works as expected for UpperTriangular matrices.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 10
#     A = randn(rng, P, P)
#     Σ = A'A + 1e-6I

#     f = Σ->logdet(chol(Symmetric(Σ)))
#     @test all(Zygote.gradient(f, Σ)[1] .- FDM.grad(fdm, f, Σ) .< 1e-8)
# end

# # Check that addition of matrices and uniform scalings works as hoped.
# let
#     fdm = central_fdm(5, 1)
#     rng, P = MersenneTwister(123456), 10
#     A = randn(rng, P, P)

#     f = A->sum(A + 5I)
#     @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
# end

# # EQ evaluation.
# let
#     fdm, rng = central_fdm(5, 1), MersenneTwister(123456)

#     x, x′ = randn(rng), randn(rng)
#     f_eval_1 = x->EQ()(x, x′)
#     @test abs(Zygote.gradient(f_eval_1, x)[1] - fdm(f_eval_1, x)) < 1e-8
#     f_eval_2 = x′->EQ()(x, x′)
#     @test abs(Zygote.gradient(f_eval_2, x′)[1] - fdm(f_eval_2, x′)) < 1e-8
# end

# function eq_check_op_gradients(fdm, op, x, x′)

#     # Check lhs argument.
#     grad_ad_lhs = Zygote.gradient(x->sum(op(EQ(), x, x′)), x)[1]
#     grad_fdm_lhs = FDM.grad(fdm, x->sum(op(EQ(), x, x′)), x)
#     @test all(abs.(grad_ad_lhs .- grad_fdm_lhs) .< 1e-8)

#     # Check rhs argument.
#     grad_ad_rhs = Zygote.gradient(x′->sum(op(EQ(), x, x′)), x′)[1]
#     grad_fdm_rhs = FDM.grad(fdm, x′->sum(op(EQ(), x, x′)), x′)
#     @test all(abs.(grad_ad_rhs .- grad_fdm_rhs) .< 1e-8)
# end

# # Operations over EQ.
# let
#     fdm = central_fdm(5, 1)
#     rng, P, D = MersenneTwister(123456), 10, 2

#     # Irregularly-spaced scalar inputs.
#     x, x′ = randn(rng, P), randn(rng, P)
#     eq_check_op_gradients(fdm, map, x, x′)
#     eq_check_op_gradients(fdm, pairwise, x, x′)

#     # Regularly-spaced scalar inputs.
#     δ = randn(rng)
#     x = range(randn(rng), step=δ, length=P)
#     x′ = range(randn(rng), step=δ, length=P)
#     eq_check_op_gradients(fdm, map, x, x′)
#     eq_check_op_gradients(fdm, pairwise, x, x′)
# end

end
