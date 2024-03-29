# test initilization functions

@testset "initialization" begin
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

        F = svd(X)
        W1, H1 = NMF.nndsvd(X, 5; initdata=F)
        F = svd(2X)
        W2, H2 = NMF.nndsvd(2X, 5; initdata=F)
        @test W2 ≈ sqrt(T(2)) * W1 && H2 ≈ sqrt(T(2)) * H1

        W, H = NMF.nndsvd(X, 5; variant=:ar)
        @test all(W .> zero(T))
        # NMF.printf_mat(W)
    end
end