# some tests for ALSPGrad

using NMF
using Base.Test

# data

srand(5678)

p = 5
n = 8
k = 3

Wg = max(rand(p, k) - 0.3, 0.0)
Hg = max(rand(k, n) - 0.3, 0.0)
X = Wg * Hg

# test update of H

H = rand(k, n)
NMF.alspgrad_updateh!(X, Wg, H; maxiter=200)
@test_approx_eq_eps H Hg 1.0e-4

# test update of W

W = rand(p, k)
NMF.alspgrad_updatew!(X, W, Hg; maxiter=200)
@test_approx_eq_eps W Wg 1.0e-4

