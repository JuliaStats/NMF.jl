module NMF
    using StatsBase
    using Statistics
    using Printf
    using LinearAlgebra

    export nnmf

    include("common.jl")
    include("utils.jl")

    include("initialization.jl")
    include("multupd.jl")
    include("projals.jl")
    include("alspgrad.jl")
    include("coorddesc.jl")

    include("interf.jl")

end # module
