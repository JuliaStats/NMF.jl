# Successive Projection Algorithm (SPA) for separable NMF
#
#   Reference: N. Gillis and S. A. Vavasis, "Fast and robust recursive
#   algorithms for separable nonnegative matrix factorization," 
#   IEEE Transactions on Pattern Analysis and Machine Intelligence, 
#   vol. 36, no. 4, pp. 698-714, 2013. 

"""
    SPA(; obj=:mse)

Successive Projection Algorithm for separable NMF (Gillis & Vavasis). Intended
for use with `W` and `H` produced by [`spa`](@ref) initialization: passing this
to [`solve!`](@ref) assembles the [`Result`](@ref) and its objective value
without further iterating.

`obj` selects the reported objective: `:mse` (mean squared error) or `:div`
(divergence).

Reference: N. Gillis and S. A. Vavasis, "Fast and robust recursive algorithms
for separable nonnegative matrix factorization," IEEE Transactions on Pattern
Analysis and Machine Intelligence, 36(4):698-714, 2014.

# Examples

```jldoctest
julia> X = rand(8, 6);

julia> W, H = NMF.spa(X, 3);

julia> r = NMF.solve!(NMF.SPA(obj=:mse), X, W, H);

julia> size(r.W), size(r.H)
((8, 3), (3, 6))
```
"""
struct SPA <: AbstractNMFAlgorithm
    obj::Symbol   # objective :mse or :div

    function SPA(;obj=:mse)
        obj == :mse || obj == :div || throw(ArgumentError("Invalid value for obj."))
        new(obj)
    end
end

# initialization
"""
    spa(X, k; nnls_alg=(:pivot, :cache)) -> (W, H)

Initialize the rank-`k` factors of `X` with the Successive Projection Algorithm
(SPA) for separable NMF. For `X` of size `(p, n)`, returns `W` of size `(p, k)`
and `H` of size `(k, n)`. `W` collects `k` "anchor" columns of `X` selected by
successive orthogonal projection, and `H` is obtained by non-negative least
squares.

Every column of `X` must have a nonzero sum, since the columns are normalized to
sum to one before anchor selection. `nnls_alg` is the `(alg, variant)` pair
forwarded to `NonNegLeastSquares.nonneg_lsq` when solving for `H`.

Reference: N. Gillis and S. A. Vavasis, "Fast and robust recursive algorithms
for separable nonnegative matrix factorization," IEEE Transactions on Pattern
Analysis and Machine Intelligence, 36(4):698-714, 2014.

# Examples

```jldoctest
julia> X = rand(10, 7);

julia> W, H = NMF.spa(X, 3);

julia> size(W), size(H)
((10, 3), (3, 7))
```
"""
function spa(X::AbstractMatrix{T}, k::Integer; nnls_alg::Tuple{Symbol, Symbol}=(:pivot, :cache)) where T

    Base.require_one_based_indexing(X)

    # Normalize data so that columns of X sum to one. An all-zero column has no
    # well-defined normalization (0/0), so reject it rather than propagate NaNs
    # into the anchor-selection argmax.
    colsums = sum(X, dims=1)
    zerocols = findall(iszero, vec(colsums))
    isempty(zerocols) || throw(ArgumentError("spa requires every column of X to have a nonzero sum; column(s) $zerocols sum to zero."))
    R = X ./ colsums

    # W = R[:,ai], where ai are the "anchor indices"
    # (ai forms the convex hull of columns in R)
    ai = Vector{Int}(undef, k)

    # Add columns of X that are furthest from span(W)
    for j = 1:k
        # Add column with the largest residual
        ai[j] = argmax(vec(sum(R.^2, dims=1)))
        
        # Project R onto the selected column
        p = R[:,ai[j]]         	# column we're projecting on
        R -= p*(p'*R) ./(p'*p) 	# new residual matrix
    end
    
    # Estimate W as the anchor columns of X
    W = X[:,ai]
    
    # Estimate H by non-negative least squares: minimize ||X - W*H||
    H = nonneg_lsq(W, X; alg=nnls_alg[1], variant=nnls_alg[2])
    projectnn!(H) 

    return W, H
end

# calculate statistics for result
function solve!(alg::SPA, X, W, H; io::IO=stdout, rng::AbstractRNG=default_rng())
    Base.require_one_based_indexing(X, W, H)
    T = eltype(W)
    if alg.obj == :mse
        objv = convert(T, 0.5) * sqL2dist(X, W*H)
    elseif alg.obj == :div
        objv = gkldiv(X, W*H)
    else
        error("Invalid value for obj.")
    end
    return Result{T}(W, H, 0, true, objv)
end

