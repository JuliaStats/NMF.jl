using NMF

p = 5
n = 8
k = 3

for T in (Float64, Float32)
    Wg = max.(rand(T, p, k) .- T(0.5), zero(T))
    Hg = max.(rand(T, k, n) .- T(0.5), zero(T))
    X = Wg * Hg
    H = rand(T, k, n)
    NMF.solve!(NMF.CoordinateDescent{T}(), X, Wg, H)
    display(X)
    print("\n")
    display(Wg * H)

end
