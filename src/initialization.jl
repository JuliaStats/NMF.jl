# random initialization

function randinit(nrows::Integer, ncols::Integer, k::Integer; normalize::Bool=false, zeroh::Bool=false)
    W = rand(nrows, k)
    if normalize
        normalize1_cols!(W)
    end

    H = zeroh ? zeros(k, ncols) : rand(k, ncols)
    return (W, H)::(Matrix{Float64}, Matrix{Float64})
end

function randinit(X::Matrix{Float64}, k::Integer; normalize::Bool=false, zeroh::Bool=false)
    m, n = size(X)
    randinit(m, n, k; normalize=normalize, zeroh=zeroh)
end
# NNDSVD: Non-Negative Double Singular Value Decomposition
#
# Reference
# ----------
#   C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head
#   start for nonnegative matrix factorization. Pattern Recognition, 2007.
#  
function _nndsvd!(X::Matrix{Float64}, 
                  W::ContiguousMatrix{Float64}, 
                  Ht::ContiguousMatrix{Float64},
                  inith::Bool, 
                  variant::Int)

    p, n = size(X)
    k = size(W, 2)

    # compute SVD
    (U, s, V) = svd(X, thin=true)

    # main loop
    v0::Float64 = variant == 0 ? 0.0 :
                  variant == 1 ? mean(X) : mean(X) * 0.01

    for j = 1:k
        x = view(U,:,j)
        y = view(V,:,j)
        xpnrm, xnnrm = posnegnorm(x)
        ypnrm, ynnrm = posnegnorm(y)
        mp = xpnrm * ypnrm
        mn = xnnrm * ynnrm

        vj = v0
        if variant == 2
            vj *= rand()
        end

        if inith
            if mp >= mn
                scalepos!(view(W,:,j), x, 1.0 / xpnrm, vj)
                scalepos!(view(Ht,:,j), y, s[j] * mp / ypnrm, vj)
            else
                ss = sqrt(s[j] * mn)
                scaleneg!(view(W,:,j), x, 1.0 / xnnrm, vj)
                scaleneg!(view(Ht,:,j), y, s[j] * mn / ynnrm, vj)
            end            
        else
            if mp >= mn
                scalepos!(view(W,:,j), x, 1.0 / xpnrm, vj)
            else
                scaleneg!(view(W,:,j), x, 1.0 / xnnrm, vj)
            end
        end
    end
end

function nndsvd(X::Matrix{Float64}, k::Integer; 
                zeroh::Bool=false, 
                variant::Symbol=:std)

    p, n = size(X)
    ivar = variant == :std ? 0 :
           variant == :a   ? 1 :
           variant == :ar  ? 2 :
           error("Invalid value for variant")

    W = Array(Float64, p, k)
    H = Array(Float64, k, n)
    if zeroh
        Ht = contiguous_view(H, (n, k))
        _nndsvd!(X, W, Ht, false, ivar)
        fill!(H, 0.0)
    else
        Ht = Array(Float64, n, k)
        _nndsvd!(X, W, Ht, true, ivar)
        for j = 1:k
            for i = 1:n
                H[j,i] = Ht[i,j]
            end
        end
    end
    return (W, H)::(Matrix{Float64}, Matrix{Float64})
end

function posnegnorm(x::ContiguousArray{Float64})
    pn = 0.0
    nn = 0.0
    for i = 1:length(x)
        @inbounds xi = x[i]
        if xi > 0.0
            pn += abs2(xi)
        else
            nn += abs2(xi)
        end
    end
    return (sqrt(pn), sqrt(nn))::(Float64, Float64)
end

function scalepos!(y::ContiguousArray{Float64}, x::ContiguousArray{Float64}, c::Float64, v0::Float64)
    @inbounds for i = 1:length(y)
        xi = x[i]
        if xi > 0.0
            y[i] = xi * c
        else
            y[i] = v0
        end
    end
end

function scaleneg!(y::ContiguousArray{Float64}, x::ContiguousArray{Float64}, c::Float64, v0::Float64)
    @inbounds for i = 1:length(y)
        xi = x[i]
        if xi < 0.0
            y[i] = - (xi * c)
        else
            y[i] = v0
        end
    end
end

