# some tests for spa

# data
@testset "spa" begin
    p = 15
    n = 8
    k = 2

    # Matrices
    for T in (Float64, Float32)
        # Initialization test
        ϵ = eps(T)^(1/4)
        Wg = max.(rand(T, p, k) .- T(0.3), T(ϵ))
        Hg = max.(rand(T, k, n) .- T(0.3), T(ϵ))
        X = Wg * Hg
        w, h = NMF.spa(X, k)
        x = w * h
        @test all(w .>= zero(T))
        @test all(h .>= zero(T))
        @test x ≈ X atol=10.0*ϵ
        #println("ϵ =",ϵ," while ||x .- X||= ",maximum(abs.(x .- X)))

        # Separability test
        Wg, Hg = NMF.separable_data(p, n, k)
        X = Wg * Hg
        w, h = NMF.spa(X, k)
        x = w * h
        @test all(w .>= zero(T))
        @test all(h .>= zero(T))
        @test sqL2dist(X, x) < eps(T)
    end
end
