using NMF
using Test

p = 5
n = 8
k = 3

for T in (Float64, Float32)
    Wg = max.(rand(T, p, k) .- T(0.5), zero(T))
    Hg = max.(rand(T, k, n) .- T(0.5), zero(T))
    X = Wg * Hg
    W = Wg .+ rand(T, p, k)*T(0.1)

    NMF.solve!(NMF.CoordinateDescent{T}(α=0.0, maxiter=1000, tol=1e-9), X, W, Hg)

    @test X ≈ W * Hg atol=1e-4
end
