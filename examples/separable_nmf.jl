# Separable NMF

using NMF
using Printf

m, n, k = 7, 8, 3
W, H = NMF.separable_data(m, n, k)
println("Separable matrix X = ")
NMF.printf_mat(W*H)
println()

# Separable NMF
r = nnmf(W*H, k; alg=:spa, init=:spa)

# Results
println("Product of W and H = ")
NMF.printf_mat(r.W*r.H)
println()

println("Matrix W = ")
NMF.printf_mat(r.W)
println()

@printf("objvalue  = %.6e\n", r.objvalue)