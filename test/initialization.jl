# test initilization functions

for T in (Float64, Float32)

    X = rand(T, 8, 12)

    # randinit

    W, H = NMF.randinit(X, 5)
    @test size(W) == (8, 5)
    @test size(H) == (5, 12)
    @test all(W .>= zero(T))
    @test all(H .>= zero(T))

    W, H = NMF.randinit(X, 5; zeroh=true)
    @test size(W) == (8, 5)
    @test size(H) == (5, 12)
    @test all(W .>= zero(T))
    @test all(H .== zero(T))

    W, H = NMF.randinit(X, 5; normalize=true)
    @test size(W) == (8, 5)
    @test size(H) == (5, 12)
    @test all(W .>= zero(T))
    @test all(H .>= zero(T))
    @test vec(sum(W, dims=1)) ≈ ones(5)

    ## nndsvd

    Random.seed!(5678)
    W, H = NMF.nndsvd(X, 5)
    @test size(W) == (8, 5)
    @test size(H) == (5, 12)
    @test all(W .>= zero(T))
    @test all(H .>= zero(T))

    Random.seed!(5678)
    W2, H2 = NMF.nndsvd(X, 5; zeroh=true)
    @test size(W) == (8, 5)
    @test size(H) == (5, 12)
    @test all(W2 .== W)
    @test all(H2 .== zero(T))

    Random.seed!(5678)
    p, n = size(X)
    T = eltype(X)
    (U, s, V) = rsvd(X, 5)
    W3 = Matrix{T}(undef, p, 5)
    H3t = Matrix{T}(undef, n, 5)
    NMF._nndsvd!(U, s, V, X, W3, H3t, true, 0)
    @test size(W3) == (8, 5)
    @test size(H3t') == (5, 12)
    @test all(W3 .>= zero(T))
    @test all(H3t .>= zero(T))

    W, H = NMF.nndsvd(X, 5; variant=:ar)
    @test all(W .> zero(T))
    # NMF.printf_mat(W)
end