# some tests for GreedyCD

p = 5
n = 8
k = 3

Random.seed!(1234)

for T in (Float64, Float32)
    for lambda_w in (0.0, 1e-4)
        for lambda_h in (0.0, 1e-4)
            Wg = max.(rand(T, p, k) .- T(0.5), zero(T))
            Hg = max.(rand(T, k, n) .- T(0.5), zero(T))
            X = Wg * Hg
            W = Wg .+ rand(T, p, k) * T(0.1)

            NMF.solve!(NMF.GreedyCD{T}(maxiter=1000, tol=1e-9, lambda_w=lambda_w, lambda_h=lambda_h), X, W, Hg)

            @test all(W .>= zero(T))
            @test all(Hg .>= zero(T))
            @test !any(isnan.(W)) 
            @test !any(isnan.(Hg)) 
            @test X ≈ W * Hg atol = 1e-3
        end
    end
end

