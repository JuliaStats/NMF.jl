"""
NMF.jl provides algorithms and initializations for non-negative matrix
factorization: approximating a non-negative matrix `X` of size `p×n` by a
product `W*H`, where `W` is `p×k`, `H` is `k×n`, both are non-negative, and `k`
is a chosen rank.

The high-level entry point is [`nnmf`](@ref), which performs initialization and
optimization in a single call. For finer control, initialize `W` and `H` with
[`NMF.randinit`](@ref), [`NMF.nndsvd`](@ref), or [`NMF.spa`](@ref) and then run
an algorithm in place with [`NMF.solve!`](@ref).
"""
module NMF
    using StatsBase: gkldiv, sqL2dist
    using Statistics: mean
    using Printf: @printf
    using LinearAlgebra: Hermitian, cholesky!, diagind, dot, ldiv!, mul!, norm, qr!, rmul!, svd!
    using NonNegLeastSquares: nonneg_lsq
    using Random: AbstractRNG, default_rng, randperm, shuffle

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
