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

        # printing test
        redirect_stdout(devnull) do
            ret = NMF.nnmf(X, k, alg=:cd, init=:nndsvd, verbose=true)
        end
    end
end

@testset "Result show" begin
    r = NMF.Result{Float64}(ones(5, 2), ones(2, 8), 42, true, 1.5)
    str = sprint(show, r)
    @test occursin("Result{Float64}", str)
    @test occursin("5×8", str)        # X dimensions
    @test occursin("niters=42", str)
    @test occursin("converged=true", str)
    @test occursin("objvalue=1.5", str)
    # compact: does not dump the factor matrices
    @test !occursin('\n', str)
end

@static if VERSION >= v"1.11"
    @testset "public bindings" begin
        for name in (:AbstractNMFAlgorithm, :Result, :solve!, :randinit, :nndsvd,
                     :spa, :MultUpdate, :ProjectedALS, :ALSPGrad,
                     :CoordinateDescent, :GreedyCD, :SPA)
            @test Base.ispublic(NMF, name)
        end
    end
end
