using NMF
using Base.Test

tests = ["utils",
         "initialization",
         "alspgrad",
	 "spa"]

println("Running tests:")
for t in tests
    tp = "$t.jl"
    println("* $tp ...")
    include(tp)
end
