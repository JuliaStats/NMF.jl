# The factorizations are built from dense linear algebra and returned as plain
# `Matrix`es, so offset axes cannot be honored. Every public entry point that
# takes array data must reject offset input at the boundary rather than return a
# result whose axes silently disagree with the input. Without the guards,
# `nnmf` returns an all-NaN "converged" factorization and `nndsvd` a silently
# wrong one.
@testset "generic axes" begin
    X = abs.(randn(MersenneTwister(1), 8, 6)) .+ 0.1
    k = 3
    msg = "offset arrays are not supported"

    @testset "offset axes are rejected" begin
        # both data axes shifted, and each one alone
        for offset in ((-2, -2), (-2, 0), (0, -2))
            Xo = OffsetArray(X, offset...)
            for alg in (:greedycd, :cd, :multmse, :multdiv, :projals, :alspgrad)
                @test_throws msg nnmf(Xo, k; alg)
            end
            @test_throws msg nnmf(Xo, k; alg=:spa, init=:spa)
            @test_throws msg NMF.randinit(Xo, k)
            @test_throws msg NMF.nndsvd(Xo, k)
            @test_throws msg NMF.spa(Xo, k)
        end

        # :custom initialization must reject offset W0/H0 too
        Wo = OffsetArray(rand(8, k), -2, 0)
        Ho = OffsetArray(rand(k, 6), 0, -2)
        @test_throws msg nnmf(X, k; init=:custom, W0=Wo, H0=rand(k, 6))
        @test_throws msg nnmf(X, k; init=:custom, W0=rand(8, k), H0=Ho)
    end

    # solve! is public and guards independently of nnmf, on each of X, W, and H
    @testset "solve! rejects offset axes" begin
        W, H = NMF.randinit(X, k; normalize=true)
        Wo, Ho, Xo = OffsetArray(copy(W), -1, 0), OffsetArray(copy(H), 0, -1), OffsetArray(X, -1, -1)
        for alg in (NMF.GreedyCD{Float64}(), NMF.CoordinateDescent{Float64}(),
                    NMF.MultUpdate{Float64}(), NMF.ProjectedALS{Float64}(),
                    NMF.ALSPGrad{Float64}(), NMF.SPA())
            @test_throws msg NMF.solve!(alg, Xo, W, H)
            @test_throws msg NMF.solve!(alg, X, Wo, H)
            @test_throws msg NMF.solve!(alg, X, W, Ho)
        end
    end
end
