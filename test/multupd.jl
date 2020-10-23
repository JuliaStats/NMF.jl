using NMF
using Test

p = 5
n = 8
k = 3

Random.seed!(5678)

for T in (Float64, Float32)
    for alg in (:mse, :div)
        for lambda_w in (0.0, 1e-4)
            for lambda_h in (0.0, 1e-4)
                Wg = max.(rand(T, p, k) .- T(0.5), zero(T))
                Hg = max.(rand(T, k, n) .- T(0.5), zero(T))
                X = Wg * Hg
                W = Wg .+ rand(T, p, k)*T(0.1)

                NMF.solve!(NMF.MultUpdate{T}(obj=alg, maxiter=5000, tol=1e-9, lambda_w=lambda_w, lambda_h=lambda_h), X, W, Hg)
                
                @test all(W .>= 0.0)
                @test all(Hg .>= 0.0)
                @test !any(isnan.(W)) 
                @test !any(isnan.(Hg)) 
                @test X ≈ W * Hg atol=1e-2
            end
        end
    end
end