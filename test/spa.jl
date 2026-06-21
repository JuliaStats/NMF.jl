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
        Wg, Hg = separable_data(p, n, k)
        X = Wg * Hg
        w, h = NMF.spa(X, k)
        x = w * h
        @test all(w .>= zero(T))
        @test all(h .>= zero(T))
        @test sqL2dist(X, x) < eps(T)
    end

    # SPA as a factorization algorithm: constructs without a type parameter
    # and produces a Result whose element type follows the factors.
    for T in (Float64, Float32)
        Wg, Hg = separable_data(p, n, k)
        X = T.(Wg * Hg)
        W, H = NMF.spa(X, k)
        ret = NMF.solve!(NMF.SPA(obj=:mse), X, W, H)
        @test ret isa NMF.Result{T}
        @test ret.converged
    end
    @test_throws ArgumentError NMF.SPA(obj=:nonsense)
end

@testset "algorithm type hierarchy" begin
    for A in (NMF.MultUpdate{Float64}, NMF.ProjectedALS{Float64},
              NMF.ALSPGrad{Float64}, NMF.CoordinateDescent{Float64},
              NMF.GreedyCD{Float64}, NMF.SPA)
        @test A <: NMF.AbstractNMFAlgorithm
    end
end
