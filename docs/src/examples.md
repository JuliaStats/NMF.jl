```@meta
CurrentModule = NMF
```

# Examples

The examples below construct an exact rank-4 non-negative matrix `X = W0*H0` so
the factorization can recover it closely; the reconstruction error `‖W*H − X‖ /
‖X‖` is therefore small. Real data rarely admits an exact low-rank
factorization, so expect larger residuals in practice.

## Using the high-level function

Pass the algorithm through the `alg` keyword and read the factors off the
[`Result`](@ref):

```jldoctest examples
julia> using NMF, Random, LinearAlgebra

julia> Random.seed!(1234);

julia> W0 = abs.(randn(20, 4)); H0 = abs.(randn(4, 50));

julia> X = W0 * H0;

julia> r = nnmf(X, 4; alg=:cd, maxiter=500, tol=1.0e-8);

julia> size(r.W), size(r.H)
((20, 4), (4, 50))

julia> norm(r.W * r.H - X) / norm(X) < 0.05
true
```

## Initializing and solving separately

For finer control, build the initial factors with an initializer and refine them
in place with [`solve!`](@ref). Algorithms that update both factors (here
[`GreedyCD`](@ref)) need both `W` and `H` initialized:

```jldoctest examples
julia> W, H = NMF.nndsvd(X, 4; variant=:ar);

julia> r = NMF.solve!(NMF.GreedyCD{Float64}(maxiter=200), X, W, H);

julia> norm(r.W * r.H - X) / norm(X) < 0.05
true
```

Algorithms that update only `W`, such as [`ProjectedALS`](@ref), can start from a
zero `H` via `zeroh=true`:

```jldoctest examples
julia> W, H = NMF.randinit(X, 4; zeroh=true);

julia> r = NMF.solve!(NMF.ProjectedALS{Float64}(maxiter=200), X, W, H);

julia> norm(r.W * r.H - X) / norm(X) < 0.05
true
```

## Separable factorization

When `X` is separable — its columns lie in the cone spanned by a subset of its
own columns — the successive projection algorithm recovers the factors directly.
Select it with `alg=:spa`, which requires `init=:spa`:

```jldoctest examples
julia> Random.seed!(0);

julia> Ws = abs.(randn(10, 3));

julia> V = rand(3, 5); V ./= sum(V, dims=1);   # each column sums to 1

julia> Xs = Ws * hcat(Matrix(I, 3, 3), V);     # a separable matrix

julia> r = nnmf(Xs, 3; alg=:spa, init=:spa);

julia> norm(r.W * r.H - Xs) / norm(Xs) < 1.0e-6
true
```

## Regularization

Some algorithms accept regularization coefficients. For example,
[`CoordinateDescent`](@ref) mixes L1 and L2 penalties through `α` and `l₁ratio`:

```jldoctest examples
julia> W, H = NMF.nndsvd(X, 4; variant=:ar);

julia> r = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=200, α=0.5, l₁ratio=0.5), X, W, H);

julia> size(r.W), size(r.H)
((20, 4), (4, 50))
```
