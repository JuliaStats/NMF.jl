# some tests for MultUpdate

@testset "multupd" begin
    for T in (Float64, Float32)
        for alg in (:mse, :div)
            for lambda_w in (0.0, 1e-4)
                for lambda_h in (0.0, 1e-4)
                    X, Wg, Hg = laurberg6x3(T(0.3))
                    W = Wg .+ rand(T, size(Wg)...)*T(0.1)

                    NMF.solve!(NMF.MultUpdate{T}(obj=alg, maxiter=5000, tol=1e-9, lambda_w=lambda_w, lambda_h=lambda_h), X, W, Hg)

                    @test all(W .>= zero(T))
                    @test all(Hg .>= zero(T))
                    @test !any(isnan.(W))
                    @test !any(isnan.(Hg))
                    @test X ≈ W * Hg atol=1e-2
                end
            end
        end
    end
end
