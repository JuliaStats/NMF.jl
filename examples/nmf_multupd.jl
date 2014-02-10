# NMF using Multiplicative update

using NMF

# data
p = 3
k = 2
n = 5

Wg = abs(randn(p, k))
Hg = abs(randn(k, n))
X = Wg * Hg #+ 0.1 * randn(p, n)

# randomly perturb the ground-truth
# to provide a reasonable init
#
# in practice, one have to resort to
# other methods to initialize W & H
#
W0 = Wg .* (0.8 + 0.4 * rand(size(Wg)))
H0 = Hg .* (0.8 + 0.4 * rand(size(Hg)))


# run algorithm (using MSE obj)

println("obj: minimize MSE")
println("---------------------------")

alg = NMFMultUpdate(obj=:mse, maxiter=20, verbose=true)
r = nmf_solve!(alg, X, copy(W0), copy(H0))

println("numiters  = $(r.niters)")
println("converged = $(r.converged)")
@printf("objvalue  = %.6e\n", r.objvalue)
println()

# run algorithm (using divergence obj)

println("obj: minimize divergence")
println("---------------------------")

alg = NMFMultUpdate(obj=:div, maxiter=20, verbose=true)
r = nmf_solve!(alg, X, copy(W0), copy(H0))

println("numiters  = $(r.niters)")
println("converged = $(r.converged)")
@printf("objvalue  = %.6e\n", r.objvalue)
println()

