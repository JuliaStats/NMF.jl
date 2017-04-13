# some tests for spa

# data

srand(5678)

p = 5
n = 8
k = 3

# Matrices

    T=Float64
    Wg = max.(rand(T, p, k) .- 0.3, 0)
    Hg = max.(rand(T, k, n) .- 0.3, 0)
    X = Wg * Hg

    #H = rand(T, k, n)

    w,h=NMF.spa(X, k)
    x=w*h
    @test all(h+eps(T)^(1/4) .>= 0.0)
    @test x ≈ X atol=eps(T) ^(1 / 4)


