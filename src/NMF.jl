module NMF

	using MLBase

	export 
	nmf_solve!,
	NMFResult, NMFMultUpdate


    include("common.jl")
    include("multupd.jl")

end # module
