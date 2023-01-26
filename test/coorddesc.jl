# some tests for CoordinateDescent

@testset "coorddesc" begin
    for T in (Float64, Float32)
        X, Wg, Hg = laurberg6x3(T(0.3))
        W = Wg .+ rand(T, size(Wg)...)*T(0.1)
        NMF.solve!(NMF.CoordinateDescent{T}(α=0.0, maxiter=1000, tol=1e-9), X, W, Hg)
        @test X ≈ W * Hg atol=1e-4

        # Regularization
        X, Wg, Hg = laurberg6x3(T(0.3))
        W = Wg .+ rand(T, size(Wg)...)*T(0.1)
        NMF.solve!(NMF.CoordinateDescent{T}(α=1e-4, l₁ratio=0.5, shuffle=true, maxiter=1000, tol=1e-9), X, W, Hg)
        @test X ≈ W * Hg atol=1e-2
    end
end
