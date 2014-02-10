# NMF using Multiplicative update

using NMF

# data
p = 8
k = 5
n = 100

Wg = abs(randn(p, k))
Hg = abs(randn(k, n))
X = Wg * Hg + 0.1 * randn(p, n)

# run algorithm

# randomly perturb the ground-truth
# to provide a reasonable init
#
# in practice, one have to resort to
# other methods to initialize W & H
#
W = Wg .* (0.8 + 0.4 * rand(size(Wg)))
H = Hg .* (0.8 + 0.4 * rand(size(Hg)))

alg = NMFMultUpdate(verbose=true, maxiter=20)
r = nmf_solve!(alg, X, W, H)

println("numiters  = $(r.niters)")
println("converged = $(r.converged)")
println("objvalue  = $(r.objvalue)")
