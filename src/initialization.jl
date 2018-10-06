
# random initialization

function randinit(X, k::Integer; normalize::Bool=false, zeroh::Bool=false)
    p, n = size(X)
    T = eltype(X)

    W = rand(T, p, k)
    if normalize
        normalize1_cols!(W)
    end

    H = zeroh ? zeros(T, k, n) : rand(T, k, n)
    return W, H
end


# NNDSVD: Non-Negative Double Singular Value Decomposition
#
# Reference
# ----------
#   C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head
#   start for nonnegative matrix factorization. Pattern Recognition, 2007.
#
function _nndsvd!(X, W, Ht, inith::Bool, variant::Int)

    p, n = size(X)
    k = size(W, 2)
    T = eltype(W)

    # compute SVD
    (U, s, V) = svd(X, full=false)

    # main loop
    v0 = variant == 0 ? zero(T) :
         variant == 1 ? convert(T, mean(X)) : convert(T, mean(X) * 0.01)

    for j = 1:k
        x = view(U,:,j)
        y = view(V,:,j)
        xpnrm, xnnrm = posnegnorm(x)
        ypnrm, ynnrm = posnegnorm(y)
        mp = xpnrm * ypnrm
        mn = xnnrm * ynnrm

        vj = v0
        if variant == 2
            vj *= rand(T)
        end

        if inith
            if mp >= mn
                scalepos!(view(W,:,j), x, 1 / xpnrm, vj)
                scalepos!(view(Ht,:,j), y, s[j] * mp / ypnrm, vj)
            else
                ss = sqrt(s[j] * mn)
                scaleneg!(view(W,:,j), x, 1 / xnnrm, vj)
                scaleneg!(view(Ht,:,j), y, s[j] * mn / ynnrm, vj)
            end
        else
            if mp >= mn
                scalepos!(view(W,:,j), x, 1 / xpnrm, vj)
            else
                scaleneg!(view(W,:,j), x, 1 / xnnrm, vj)
            end
        end
    end
end

function nndsvd(X, k::Integer; zeroh::Bool=false, variant::Symbol=:std)

    p, n = size(X)
    T = eltype(X)
    ivar = variant == :std ? 0 :
           variant == :a   ? 1 :
           variant == :ar  ? 2 :
           throw(ArgumentError("Invalid value for variant"))

    W = Matrix{T}(undef, p, k)
    H = Matrix{T}(undef, k, n)
    if zeroh
        Ht = reshape(view(H,:,:), (n, k))
        _nndsvd!(X, W, Ht, false, ivar)
        fill!(H, 0)
    else
        Ht = Matrix{T}(undef, n, k)
        _nndsvd!(X, W, Ht, true, ivar)
        for j = 1:k
            for i = 1:n
                H[j,i] = Ht[i,j]
            end
        end
    end
    return (W, H)
end

function posnegnorm(x::AbstractArray{T}) where T
    pn = zero(T)
    nn = zero(T)
    for i = 1:length(x)
        @inbounds xi = x[i]
        if xi > zero(T)
            pn += abs2(xi)
        else
            nn += abs2(xi)
        end
    end
    return (sqrt(pn), sqrt(nn))
end

function scalepos!(y, x, c::T, v0::T) where T<:Number
    @inbounds for i = 1:length(y)
        xi = x[i]
        if xi > zero(T)
            y[i] = xi * c
        else
            y[i] = v0
        end
    end
end

function scaleneg!(y, x, c::T, v0::T) where T<:Number
    @inbounds for i = 1:length(y)
        xi = x[i]
        if xi < zero(T)
            y[i] = - (xi * c)
        else
            y[i] = v0
        end
    end
end
