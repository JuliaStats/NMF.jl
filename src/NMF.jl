module NMF

	using MLBase

	import Base: sum!

	export 
	nmf_solve!,
	NMFResult, NMFMultUpdate


    include("common.jl")
    include("multupd.jl")

end # module
