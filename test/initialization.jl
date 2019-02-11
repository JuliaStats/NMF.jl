# test initilization functions

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
@test vec(sum(W, dims=1)) ≈ ones(5)

## nndsvd

W, H = NMF.nndsvd(X, 5)
@test size(W) == (8, 5)
@test size(H) == (5, 12)
@test all(W .>= 0.0)
@test all(H .>= 0.0)
@test vec(sum(abs2, W, dims=1)) ≈ ones(5) atol=1.0e-8

W2, H2 = NMF.nndsvd(X, 5; zeroh=true)
@test size(W) == (8, 5)
@test size(H) == (5, 12)
@test all(W2 .== W)
@test all(H2 .== 0.0)

W, H = NMF.nndsvd(X, 5; variant=:ar)
@test all(W .> 0.0)
# NMF.printf_mat(W)
