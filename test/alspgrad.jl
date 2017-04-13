# some tests for ALSPGrad

# data

srand(5678)

p = 5
n = 8
k = 3

# Matrices

T = Float64
    Wg = max.(rand(T, p, k) .- 0.3, 0)
    Hg = max.(rand(T, k, n) .- 0.3, 0)
    X = Wg * Hg

    # test update of H

    H = rand(T, k, n)
    NMF.alspgrad_updateh!(X, Wg, H; maxiter=200)
    @test all(H .>= 0.0)
    @test H ≈ Hg atol = eps(T)^(1/4)

    # test update of W

    W = rand(T, p, k)
    NMF.alspgrad_updatew!(X, W, Hg; maxiter=200)
    @test all(W .>= 0.0)
    @test W ≈ Wg atol = eps(T)^(1/4)

    NMF.solve!(NMF.ALSPGrad{T}(), X, W, H)

