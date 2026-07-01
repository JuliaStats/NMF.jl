
# random initialization

function randinit(nrows::Integer, ncols::Integer, k::Integer, T::DataType;
                  normalize::Bool=false, zeroh::Bool=false, rng::AbstractRNG=default_rng())
    W = rand(rng, T, nrows, k)
    if normalize
        normalize1_cols!(W)
    end

    H = zeroh ? zeros(T, k, ncols) : rand(rng, T, k, ncols)
    return W, H
end

"""
    randinit(X, k; normalize=false, zeroh=false, rng=default_rng()) -> (W, H)

Randomly initialize the factors for a rank-`k` factorization of `X`. For `X` of
size `(p, n)`, returns `W` of size `(p, k)` and `H` of size `(k, n)` filled with
uniform random values.

If `normalize=true`, each column of `W` is scaled to sum to one. If
`zeroh=true`, `H` is returned as a zero matrix, appropriate for algorithms that
need only `W` initialized (e.g. [`ProjectedALS`](@ref)). `rng` supplies the
randomness.

# Examples

```jldoctest
julia> X = rand(10, 7);

julia> W, H = NMF.randinit(X, 3);

julia> size(W), size(H)
((10, 3), (3, 7))
```
"""
function randinit(X, k::Integer; normalize::Bool=false, zeroh::Bool=false, rng::AbstractRNG=default_rng())
    Base.require_one_based_indexing(X)
    p, n = size(X)
    randinit(p, n, k, eltype(X); normalize, zeroh, rng)
end

# NNDSVD: Non-Negative Double Singular Value Decomposition
#
# Reference
# ----------
#   C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head
#   start for nonnegative matrix factorization. Pattern Recognition, 2007.
#
function _nndsvd!(rng::AbstractRNG, U, s, V, X, W, Ht, inith::Bool, variant::Int)

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
            vj *= rand(rng, T)
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

# A randomized SVD: a basis for an approximate range of X is found by projecting
# onto a random test matrix, then the SVD is computed on that restriction. The
# only randomness is the test matrix, so `rng` makes the result reproducible.
# (Reproduces `RandomizedLinAlg.rsvd(X, k)`, which offers no way to pass an rng.)
function _rsvd(rng::AbstractRNG, X, k::Integer)
    m = size(X, 1)
    k <= m || throw(ArgumentError("Cannot find $k singular vectors of a $m-row matrix."))
    Ω = randn(rng, size(X, 2), k)
    Q = Matrix(qr!(X * Ω).Q)
    S = svd!(Q' * X)
    return (Q * S.U)[:, 1:k], S.S[1:k], (S.Vt[1:k, :])'
end

"""
    nndsvd(X, k; variant=:std, zeroh=false, initdata=nothing, rng=default_rng()) -> (W, H)

Initialize the rank-`k` factors of `X` with Non-Negative Double Singular Value
Decomposition (NNDSVD). For `X` of size `(p, n)`, returns `W` of size `(p, k)`
and `H` of size `(k, n)`.

`variant` selects the flavor: `:std` (standard, yields a sparse `W`), `:a`
(NNDSVDa), or `:ar` (NNDSVDar, recommended for dense NMF). If `zeroh=true`, `H`
is returned as a zero matrix, appropriate for algorithms that need only `W`
initialized. By default the required SVD is computed with a randomized algorithm
seeded by `rng`; pass a precomputed factorization as `initdata` (e.g.
`initdata=svd(X)`) to use it instead.

Reference: C. Boutsidis and E. Gallopoulos, "SVD based initialization: A head
start for nonnegative matrix factorization," Pattern Recognition, 2008.

# Examples

```jldoctest
julia> X = rand(10, 7);

julia> W, H = NMF.nndsvd(X, 3; variant=:ar);

julia> size(W), size(H)
((10, 3), (3, 7))
```
"""
function nndsvd(X, k::Integer; zeroh::Bool=false, variant::Symbol=:std, initdata=nothing,
                rng::AbstractRNG=default_rng())

    Base.require_one_based_indexing(X)
    p, n = size(X)
    T = eltype(X)
    ivar = variant == :std ? 0 :
           variant == :a   ? 1 :
           variant == :ar  ? 2 :
           throw(ArgumentError("Invalid value for variant"))

    U, s, V = initdata === nothing ? _rsvd(rng, X, k) : (initdata.U[:,1:k], initdata.S[1:k], initdata.V[:,1:k])

    W = Matrix{T}(undef, p, k)
    H = Matrix{T}(undef, k, n)
    if zeroh
        Ht = reshape(view(H,:,:), (n, k))
        _nndsvd!(rng, U, s, V, X, W, Ht, false, ivar)
        fill!(H, 0)
    else
        Ht = Matrix{T}(undef, n, k)
        _nndsvd!(rng, U, s, V, X, W, Ht, true, ivar)
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
