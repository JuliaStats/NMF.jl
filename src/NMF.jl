module NMF
    using StatsBase
    using Statistics
    using Printf
    using LinearAlgebra
    using NonNegLeastSquares
    using Random
    using RandomizedLinAlg

    export nnmf,
           Trace

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

    using SnoopPrecompile

    let
        @precompile_setup begin
            X = rand(8, 6)
            @precompile_all_calls begin
                for alg in (:multmse, :multdiv, :projals, :alspgrad, :cd, :greedycd)
                    for init in (:random, :nndsvd, :nndsvda, :nndsvdar, :spa)
                        nnmf(X, 4, alg=alg, init=init)
                    end
                end
            end
        end
    end
end # module
