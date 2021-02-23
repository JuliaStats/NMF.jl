module NMF
    using StatsBase
    using Statistics
    using Printf
    using LinearAlgebra
    # using NonNegLeastSquares
    using Random
    using Distributed # temporarily used

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

end # module
