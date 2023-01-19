using NMF
using Test
using Random
using LinearAlgebra
using StatsBase

tests = ["utils",
         "initialization",
         "spa",
         "multupd",
         "alspgrad",
         "coorddesc",
         "greedycd",
         "interf"]

println("Running tests:")
@testset "All tests" begin
    for t in tests
        tp = "$t.jl"
        include(tp)
    end
end
