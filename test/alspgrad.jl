# some tests for ALSPGrad

using NMF
using Base.Test

# data

srand(5678)

p = 5
n = 8
k = 3

Wg = rand(p, k)
Hg = rand(k, n)
X = Wg * Hg

# test update of H

H = rand(k, n)
NMF.alspgrad_updateh!(X, Wg, H; maxiter=200)
@test_approx_eq_eps H Hg 1.0e-4



