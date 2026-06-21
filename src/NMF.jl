module NMF
    using StatsBase: gkldiv, sqL2dist
    using Statistics: mean
    using Printf: @printf
    using LinearAlgebra: Hermitian, cholesky!, diagind, dot, ldiv!, mul!, norm, rmul!
    using NonNegLeastSquares: nonneg_lsq
    using Random: randperm, shuffle
    using RandomizedLinAlg: rsvd

    export nnmf

    include("common.jl")
    include("utils.jl")

    include("initialization.jl")
    include("spa.jl")
    include("multupd.jl")
    include("projals.jl")
    include("alspgrad.jl")
    include("coorddesc.jl")
    include("greedycd.jl")

    include("interf.jl")

    # Supported, documented bindings that are reachable via the `NMF.` prefix
    # rather than exported. `public` is a Julia 1.11+ keyword, so the eval keeps
    # this file parseable on 1.10.
    @static if VERSION >= v"1.11.0-DEV.469"
        eval(Meta.parse("public AbstractNMFAlgorithm, Result, solve!, " *
                        "randinit, nndsvd, spa, " *
                        "MultUpdate, ProjectedALS, ALSPGrad, CoordinateDescent, GreedyCD, SPA"))
    end

    using PrecompileTools: @compile_workload, @setup_workload

    let
        @setup_workload begin
            X = rand(8, 6)
            @compile_workload begin
                for alg in (:multmse, :multdiv, :projals, :alspgrad, :cd, :greedycd)
                    for init in (:random, :nndsvd, :nndsvda, :nndsvdar, :spa)
                        nnmf(X, 4, alg=alg, init=init)
                    end
                end
            end
        end
    end
end # module
