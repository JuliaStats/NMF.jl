#
# Test setup for all algorithms and initializations for both Float32 and Float64
#

srand(5678)   # Reproducable results, uncomment to get statistics

T=(Float64)   # Generate fixed datamatrix

p = 15        # Dimensionality of data vectors      
n = 18        # Number of samples 
k = 4         # Factorization constraint.

Wg = max.(rand(T, p, k) .- 0.3, 0)
Hg = max.(rand(T, k, n) .- 0.3, 0)

X = Wg * Hg   # Synthetic data matrix to factor

# Could be possible to compare directly
#niters=Int64(0)
#converged=true
#objv=0.0
#ref=NMF.Result(Wg,Hg,niters,true,objv)

println("Time       Algorithm   Init      Niters      Converge   DataType    Objv")
println("==========================================================================")                
for T in (Float32, Float64)
    for alg in (:multmse, :multdiv, :projals, :alspgrad, :spa)
        for init in (:random, :nndsvd, :nndsvda, :nndsvdar, :spa)
            t=time()
            tol=cbrt(eps(T))
            ret = NMF.nnmf(X, k, alg=alg, init=init,maxiter=5000,tol=tol)
            t=time()-t
            niters=ret.niters
            conv=ret.converged
            obj=ret.objv
            
            println(@sprintf("%9.5f",t),
                    @sprintf("  %-10s",alg),
                    @sprintf("  %-10s",init),
                    @sprintf("  %-10i",niters),
                    @sprintf("  %-7s",conv),
                    @sprintf("  %-7s",T),
                    @sprintf("%12.5f",obj)
                    )
        end
    end
end
