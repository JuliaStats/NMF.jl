# some tests for spa

# data

srand(5678)

p = 15
n = 8
k = 2

# Matrices
for T in (Float64, Float32)
    ϵ = eps(T)^(1/4)  
    Wg = max.(rand(T, p, k) .- T(0.3), ϵ)
    Hg = max.(rand(T, k, n) .- T(0.3), ϵ)
    X = Wg * Hg

    w,h=NMF.spa(X, k )
    
    x=w*h
 
    @test all(h + ϵ .>= 0.0)
    @test x ≈ X atol = 10.0*ϵ
    #println("ϵ =",ϵ," while ||x .- X||= ",maximum(abs.(x .- X)))

end


