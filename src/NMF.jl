module NMF

	using MLBase
	import Base: sum!

    include("common.jl")
    include("initialization.jl")
    include("multupd.jl")
    include("projals.jl")

end # module
