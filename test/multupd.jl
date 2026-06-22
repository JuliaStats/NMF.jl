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

    @testset "deprecated lambda keyword" begin
        # A non-negative lambda warns and forwards to lambda_w/lambda_h.
        alg = @test_logs (:warn, r"lambda is deprecated") NMF.MultUpdate{Float64}(lambda=0.5)
        @test alg.lambda_w == 0.5
        @test alg.lambda_h == 0.5
        # A negative lambda is rejected rather than silently dropped.
        @test_throws ArgumentError NMF.MultUpdate{Float64}(lambda=-1.0)
        @test_throws "lambda must be non-negative" NMF.MultUpdate{Float64}(lambda=-1.0)
    end
end
