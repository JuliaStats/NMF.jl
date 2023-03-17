@testset "interf" begin
    p = 5
    n = 8
    k = 3

    for T in (Float64, Float32)
        Wg = max.(rand(T, p, k) .- T(0.3), zero(T))
        Hg = max.(rand(T, k, n) .- T(0.3), zero(T))
        X = Wg * Hg

        for alg in (:multmse, :multdiv, :projals, :alspgrad, :cd, :greedycd)
            for init in (:random, :nndsvd, :nndsvda, :nndsvdar, :spa)
                ret = NMF.nnmf(X, k, alg=alg, init=init)
            end
        end

        for init in (:random, :nndsvd, :nndsvda, :nndsvdar, :spa)
            ret = NMF.nnmf(X, k, alg=:cd, init=init, trace=Trace{T}(save_obj=true, save_time=true))
            @test typeof(ret.trace.objective) == Vector{T}
            @test length(ret.trace.objective) == ret.niters
            @test ret.trace.objective[end] == ret.objvalue
            @test typeof(ret.trace.time) == Vector{Float64}
            @test length(ret.trace.time) == ret.niters

            ret = NMF.nnmf(X, k, alg=:cd, init=init, trace=Trace{T}(save_obj=false, save_time=true))
            @test ret.trace.objective === nothing
            @test typeof(ret.trace.time) == Vector{Float64}
            @test length(ret.trace.time) == ret.niters

            ret = NMF.nnmf(X, k, alg=:cd, init=init, trace=Trace{T}(save_obj=true, save_time=false))
            @test typeof(ret.trace.objective) == Vector{T}
            @test length(ret.trace.objective) == ret.niters
            @test ret.trace.objective[end] == ret.objvalue
            @test ret.trace.time === nothing

            ret = NMF.nnmf(X, k, alg=:cd, init=init, trace=Trace{T}(save_obj=false, save_time=false))
            @test ret.trace.objective === nothing
            @test ret.trace.time === nothing
        end

        # external initialization
        F = svd(X)
        for alg in (:multmse, :multdiv, :projals, :alspgrad, :cd, :greedycd)
            ret = NMF.nnmf(X, k, alg=alg, init=:nndsvd, initdata=F)
        end

        # replicates test
        rep = NMF.nnmf(X, k, replicates=10, maxiter=10, alg=:multmse)
        ret = NMF.nnmf(X, k, W0=rep.W, H0=rep.H, init=:custom)

        # spa test
        ret = NMF.nnmf(X, k, alg=:spa, init=:spa)

        # update_H test
        W = max.(rand(T, p, k) .- T(0.3), zero(T))
        H = max.(rand(T, k, n) .- T(0.3), zero(T))
        for alg in (:multmse, :multdiv, :projals, :alspgrad, :cd, :greedycd)
            ret = NMF.nnmf(X, k, alg=alg, init=:custom, W0=copy(W), H0=copy(H), update_H=false)
            @test all(H .== ret.H)
            @test any(W .!= ret.W)
        end
    end
end
