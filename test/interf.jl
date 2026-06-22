@testset "interf" begin
    Xg, Wg0, Hg0 = separable6x3()
    p, k = size(Wg0)
    n = size(Hg0, 2)

    for T in (Float64, Float32)
        X = T.(Xg)

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

        # update_H test: with update_H=false, H is held fixed and only W moves,
        # so start from a deliberately non-optimal W.
        W = fill(T(1), p, k)
        H = T.(Hg0)
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

@testset "Result construction" begin
    W = ones(5, 2); H = ones(2, 8)
    ref = NMF.Result{Float64}(W, H, 42, true, 1.5)

    # The inner constructor coerces non-Matrix and differently-typed arguments
    # into the declared fields.
    @test NMF.Result{Float64}(view(W, :, :), H, 42, true, 1.5) == ref
    @test NMF.Result{Float64}(Int.(W), Int.(H), 42, true, 1.5) == ref
    @test NMF.Result{Float64}(W, H, 42, true, 3//2) == ref

    # The outer constructor infers T from the factor element types.
    r = NMF.Result(W, H, 42, true, 1.5)
    @test r isa NMF.Result{Float64}
    @test r == ref
    @test NMF.Result(Float32.(W), Float32.(H), 42, true, 1.5f0) isa NMF.Result{Float32}

    @test_throws DimensionMismatch NMF.Result{Float64}(W, ones(3, 8), 42, true, 1.5)
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

@testset "constructor argument validation" begin
    # Every iterative algorithm rejects a degenerate maxiter/tol rather than
    # silently returning niters=0, converged=false.
    for Alg in (NMF.MultUpdate, NMF.ProjectedALS, NMF.ALSPGrad,
                NMF.CoordinateDescent, NMF.GreedyCD)
        @test_throws ArgumentError Alg{Float64}(maxiter=1)
        @test_throws "maxiter must be greater than 1" Alg{Float64}(maxiter=0)
        @test_throws ArgumentError Alg{Float64}(tol=0.0)
        @test_throws "tol must be positive" Alg{Float64}(tol=-1.0)
    end
end
