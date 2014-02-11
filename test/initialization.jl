# test initilization functions

import NMF
using Base.Test

X = rand(8, 12)

# randinit

W, H = NMF.randinit(X, 5)
@test size(W) == (8, 5)
@test size(H) == (5, 12)
@test all(W .>= 0.0)
@test all(H .>= 0.0)

W, H = NMF.randinit(X, 5; zeroh=true)
@test size(W) == (8, 5)
@test size(H) == (5, 12)
@test all(W .>= 0.0)
@test all(H .== 0.0)

W, H = NMF.randinit(X, 5; normalize=true)
@test size(W) == (8, 5)
@test size(H) == (5, 12)
@test all(W .>= 0.0)
@test all(H .>= 0.0)
@test_approx_eq vec(sum(W, 1)) ones(5)

