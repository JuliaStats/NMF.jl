# Using Naive ALS

using NMF

# data
p = 8
k = 5
n = 100

Wg = abs(randn(p, k))
Hg = abs(randn(k, n))
X = Wg * Hg + 0.1 * randn(p, n)

W0 = Wg .* (0.8 + 0.4 * rand(size(Wg)))
H0 = Hg .* (0.8 + 0.4 * rand(size(Hg)))

# run algorithm (using MSE obj)

println("Using Naive ALS")
println("---------------------------")

alg = NMF.NaiveALS(maxiter=30, verbose=true)
r = nmf_solve!(alg, X, copy(W0), copy(H0))

println("numiters  = $(r.niters)")
println("converged = $(r.converged)")
@printf("objvalue  = %.6e\n", r.objvalue)
println()
