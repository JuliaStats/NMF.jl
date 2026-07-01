using NMF
using Test
using Random
using LinearAlgebra
using StatsBase
using Aqua
using ExplicitImports
using OffsetArrays

include("testproblems.jl")

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
    @testset "Aqua" begin
        Aqua.test_all(NMF)
    end
    @testset "ExplicitImports" begin
        # The public-ness checks rely on `Base.ispublic`, available only on
        # Julia 1.11+; on older versions they fall back to `isexported` and
        # would false-positive on public-but-unexported bindings.
        test_explicit_imports(NMF;
                              all_explicit_imports_are_public   = VERSION >= v"1.11",
                              all_qualified_accesses_are_public = VERSION >= v"1.11")
    end
    include("genericaxes.jl")
    for t in tests
        tp = "$t.jl"
        include(tp)
    end
end
