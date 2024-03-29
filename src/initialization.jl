
# random initialization

function randinit(nrows::Integer, ncols::Integer, k::Integer, T::DataType; normalize::Bool=false, zeroh::Bool=false)
    W = rand(T, nrows, k)
    if normalize
        normalize1_cols!(W)
    end

    H = zeroh ? zeros(T, k, ncols) : rand(T, k, ncols)
    return W, H
end

function randinit(X, k::Integer; normalize::Bool=false, zeroh::Bool=false)
    p, n = size(X)
    randinit(p, n, k, eltype(X); normalize=normalize, zeroh=zeroh)
end

# NNDSVD: Non-Negative Double Singular Value Decomposition
#
# Reference
# ----------
#   C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head
#   start for nonnegative matrix factorization. Pattern Recognition, 2007.
#
function _nndsvd!(U, s, V, X, W, Ht, inith::Bool, variant::Int)

    k = size(W, 2)
    T = eltype(W)

    U = T.(U)
    s = T.(s)
    V = T.(V)

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
                ss = sqrt(s[j] * mp)
                scalepos!(view(W,:,j), x, ss / xpnrm, vj)
                scalepos!(view(Ht,:,j), y, ss / ypnrm, vj)
            else
                ss = sqrt(s[j] * mn)
                scaleneg!(view(W,:,j), x, ss / xnnrm, vj)
                scaleneg!(view(Ht,:,j), y, ss / ynnrm, vj)
            end
        else
            if mp >= mn
                ss = sqrt(s[j] * mp)
                scalepos!(view(W,:,j), x, ss / xpnrm, vj)
            else
                ss = sqrt(s[j] * mn)
                scaleneg!(view(W,:,j), x, ss / xnnrm, vj)
            end
        end
    end
end

function nndsvd(X, k::Integer; zeroh::Bool=false, variant::Symbol=:std, initdata=nothing)

    p, n = size(X)
    T = eltype(X)
    ivar = variant == :std ? 0 :
           variant == :a   ? 1 :
           variant == :ar  ? 2 :
           throw(ArgumentError("Invalid value for variant"))

    U, s, V = initdata === nothing ? rsvd(X, k) : (initdata.U[:,1:k], initdata.S[1:k], initdata.V[:,1:k])

    W = Matrix{T}(undef, p, k)
    H = Matrix{T}(undef, k, n)
    if zeroh
        Ht = reshape(view(H,:,:), (n, k))
        _nndsvd!(U, s, V, X, W, Ht, false, ivar)
        fill!(H, 0)
    else
        Ht = Matrix{T}(undef, n, k)
        _nndsvd!(U, s, V, X, W, Ht, true, ivar)
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
