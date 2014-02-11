
tests = ["utils"]

println("Running tests:")
for t in tests
	tp = joinpath("test", "$t.jl")
	println("* $tp ...")
	include(tp)
end
