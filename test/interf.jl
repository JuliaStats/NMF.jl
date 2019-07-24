Random.seed!(5678)

p = 5
n = 8
k = 3

for T in (Float64, Float32)
    Wg = max.(rand(T, p, k) .- 0.3, 0)
    Hg = max.(rand(T, k, n) .- 0.3, 0)
    X = Wg * Hg

    for alg in (:multmse, :multdiv, :projals, :alspgrad, :cd)
        for init in (:random, :nndsvd, :nndsvda, :nndsvdar)
            ret = NMF.nnmf(X, k, alg=alg, init=init)
        end
    end
end
