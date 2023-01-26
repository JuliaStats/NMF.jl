# some tests for ALSPGrad

@testset "alspgrad" begin
    # Matrices
    for T in (Float64, Float32)
        X, Wg, Hg = laurberg6x3(T(0.3))

        # test update of H

        H = rand(T, size(Hg)...)
        NMF.alspgrad_updateh!(X, Wg, H; maxiter=1000, tolg=eps(T))
        @test all(H .>= zero(T))
        @test H ≈ Hg atol=eps(T)^(1/4)

        # test update of W

        W = rand(T, size(Wg)...)
        NMF.alspgrad_updatew!(X, W, Hg; maxiter=1000, tolg=eps(T))
        @test all(W .>= zero(T))
        @test W ≈ Wg atol=eps(T)^(1/4)

        NMF.solve!(NMF.ALSPGrad{T}(), X, W, H)
    end

end
