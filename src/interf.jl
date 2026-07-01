# Interface function: nnmf

"""
    nnmf(X, k; init=:nndsvdar, alg=:greedycd, maxiter=100, tol=…, kwargs...) -> Result

Factorize the non-negative matrix `X` into a product `W*H` of two non-negative
matrices, where `W` has `k` columns and `H` has `k` rows, so that `W*H`
approximates `X`. `X` is `p×n`, `W` is `p×k`, and `H` is `k×n`; `k` must not
exceed `min(p, n)`. The factors are initialized and then optimized, and the
[`Result`](@ref) is returned.

# Keyword arguments
- `init::Symbol=:nndsvdar`: initialization method, one of `:random`, `:nndsvd`,
  `:nndsvda`, `:nndsvdar`, `:spa`, or `:custom` (supply `W0` and `H0`).
- `initdata=nothing`: for the `:nndsvd` variants, a precomputed SVD of `X`
  (e.g. `svd(X)`) to use in place of an internally computed randomized SVD.
- `alg::Symbol=:greedycd`: factorization algorithm, one of `:multmse`,
  `:multdiv`, `:projals`, `:alspgrad`, `:cd`, `:greedycd`, or `:spa`. Selecting
  `:spa` requires `init=:spa`.
- `maxiter::Integer=100`: maximum number of iterations.
- `tol::Real=cbrt(eps(eltype(X))/100)`: convergence tolerance on the relative
  change of `W` and `H`.
- `replicates::Integer=1`: number of runs from independent random
  initializations; the result with the smallest objective value is returned.
- `W0`, `H0`: custom initial factors, required (and used) only when
  `init=:custom`. They may be overwritten; pass copies to preserve them.
- `update_H::Bool=true`: if `false`, hold `H` fixed and update only `W`.
- `verbose::Bool=false`: whether to print per-iteration progress.
- `io::IO=stdout`: stream for progress output when `verbose=true`.
- `rng::AbstractRNG=default_rng()`: random number generator for initialization
  and any randomized algorithm steps.

# Examples

```jldoctest
julia> X = rand(8, 6);

julia> r = nnmf(X, 3; alg=:multmse, maxiter=50, tol=1.0e-4);

julia> size(r.W), size(r.H)
((8, 3), (3, 6))
```
"""
function nnmf(X::AbstractMatrix{T}, k::Integer;
              init::Symbol=:nndsvdar,
              initdata=nothing,
              alg::Symbol=:greedycd,
              maxiter::Integer=100,
              tol::Real=cbrt(eps(T)/100),
              replicates::Integer=1,
              W0::Union{AbstractMatrix{T}, Nothing}=nothing,
              H0::Union{AbstractMatrix{T}, Nothing}=nothing,
              update_H::Bool=true,
              verbose::Bool=false,
              io::IO=stdout,
              rng::AbstractRNG=default_rng()) where T

    # The factorization is built from dense linear algebra and returned as plain
    # `Matrix`es, so offset axes cannot be honored; reject them rather than
    # return a result whose axes silently disagree with the input.
    Base.require_one_based_indexing(X)

    eltype(X) <: Number && all(t -> t >= zero(T), X) || throw(ArgumentError("The elements of X must be non-negative."))

    p, n = size(X)
    k <= min(p, n) || throw(ArgumentError("The value of k should not exceed min(size(X))."))

    replicates >= 1 || throw(ArgumentError("The value of replicates must be positive."))

    if !update_H && init != :custom
        @warn "Only W will be updated."
    end

    if init == :custom
        W0 !== nothing && H0 !== nothing || throw(ArgumentError("To use :custom initialization, set W0 and H0."))
        Base.require_one_based_indexing(W0, H0)
        eltype(W0) <: Number && all(t -> t >= zero(T), W0) || throw(ArgumentError("The elements of W0 must be non-negative."))
        p0, k0 = size(W0)
        p == p0 && k == k0 || throw(ArgumentError("Invalid size for W0."))
        eltype(H0) <: Number && all(t -> t >= zero(T), H0) || throw(ArgumentError("The elements of H0 must be non-negative."))
        k0, n0 = size(H0)
        k == k0 && n == n0 || throw(ArgumentError("Invalid size for H0."))
    else
        W0 === nothing && H0 === nothing || @warn "Ignore W0 and H0 except for :custom initialization."
    end

    # determine whether H needs to be initialized
    initH = alg != :projals

    # perform initialization
    if init == :random
        W, H = randinit(X, k; zeroh=!initH, normalize=true, rng)
    elseif init == :nndsvd
        W, H = nndsvd(X, k; zeroh=!initH, initdata=initdata, rng)
    elseif init == :nndsvda
        W, H = nndsvd(X, k; variant=:a, zeroh=!initH, initdata=initdata, rng)
    elseif init == :nndsvdar
        W, H = nndsvd(X, k; variant=:ar, zeroh=!initH, initdata=initdata, rng)
    elseif init == :spa
        W, H = spa(X, k)
    elseif init == :custom
        W, H = W0, H0
    else
        throw(ArgumentError("Invalid value for init."))
    end
    W = W::Matrix{T}
    H = H::Matrix{T}

    # choose algorithm
    if alg == :projals
        ret = solve_replicates!(ProjectedALS{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H), X, W, H; replicates, initH, io, rng)
    elseif alg == :alspgrad
        ret = solve_replicates!(ALSPGrad{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H), X, W, H; replicates, initH, io, rng)
    elseif alg == :multmse
        ret = solve_replicates!(MultUpdate{T}(obj=:mse, maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H), X, W, H; replicates, initH, io, rng)
    elseif alg == :multdiv
        ret = solve_replicates!(MultUpdate{T}(obj=:div, maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H), X, W, H; replicates, initH, io, rng)
    elseif alg == :cd
        ret = solve_replicates!(CoordinateDescent{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H), X, W, H; replicates, initH, io, rng)
    elseif alg == :greedycd
        ret = solve_replicates!(GreedyCD{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H), X, W, H; replicates, initH, io, rng)
    elseif alg == :spa
        if init != :spa
            throw(ArgumentError("Invalid value for init, use :spa instead."))
        end
        ret = solve_replicates!(SPA(obj=:mse), X, W, H; replicates, initH, io, rng)
    else
        throw(ArgumentError("Invalid algorithm."))
    end

    return ret
end

function solve_replicates!(alginst, X, W, H; replicates, initH, io::IO=stdout, rng::AbstractRNG=default_rng())
    ret = solve!(alginst, X, W, H; io, rng)
    k = size(W, 2)

    # replicates
    minobjv = ret.objvalue
    for _ in 2:replicates
        Wrand, Hrand = randinit(X, k; zeroh=!initH, normalize=true, rng)
        tmp = solve!(alginst, X, Wrand, Hrand; io, rng)
        if minobjv > tmp.objvalue
            ret = tmp
            minobjv = tmp.objvalue
        end
    end

    return ret
end
