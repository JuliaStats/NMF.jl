#
# Multiplicative updating algorithm
#
# Reference:
#   Daniel D. Lee and H. Sebastian Seung. Algorithms for Non-negative
#   Matrix Factorization. Advances in NIPS, 2001.
#

"""
    MultUpdate{T}(; obj=:mse, maxiter=100, tol=cbrt(eps(T)), update_H=true,
                  lambda_w=zero(T), lambda_h=zero(T), verbose=false)

Multiplicative-update algorithm (Lee & Seung). Both `W` and `H` must be
initialized before [`solve!`](@ref).

`obj` selects the objective: `:mse` (mean squared error) or `:div` (divergence).
`lambda_w` and `lambda_h` are L1 regularization coefficients for `W` and `H`.
`maxiter` bounds the iterations, `tol` is the convergence tolerance on the
relative change of `W` and `H`, `update_H=false` holds `H` fixed, and
`verbose=true` prints per-iteration progress.

Reference: D. D. Lee and H. S. Seung, "Algorithms for Non-negative Matrix
Factorization," Advances in NIPS, 2001.

# Examples

```jldoctest
julia> X = rand(8, 6);

julia> W, H = NMF.randinit(X, 3);

julia> r = NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse, maxiter=50), X, W, H);

julia> size(r.W), size(r.H)
((8, 3), (3, 6))
```
"""
struct MultUpdate{T} <: AbstractNMFAlgorithm
    obj::Symbol                 # objective :mse or :div
    maxiter::Int                # maximum number of iterations
    verbose::Bool               # whether to show procedural information
    tol::T                      # change tolerance upon convergence
    update_H::Bool              # whether to update H
    lambda_w::T                 # L1 regularization coefficient for W
    lambda_h::T                 # L1 regularization coefficient for H

    function MultUpdate{T}(;obj::Symbol=:mse,
                            maxiter::Integer=100,
                            verbose::Bool=false,
                            tol::Real=cbrt(eps(T)),
                            update_H::Bool=true,
                            lambda_w::Real=zero(T),
                            lambda_h::Real=zero(T),
                            lambda::Union{Real, Nothing}=nothing) where T

        obj == :mse || obj == :div || throw(ArgumentError("Invalid value for obj."))
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        lambda_w >= 0 || throw(ArgumentError("lambda_w must be non-negative."))
        lambda_h >= 0 || throw(ArgumentError("lambda_h must be non-negative."))
        if lambda !== nothing
            lambda >= 0 || throw(ArgumentError("lambda must be non-negative."))
            @warn "lambda is deprecated, use lambda_w and lambda_h instead."
            lambda_w = iszero(lambda_w) ? lambda : lambda_w
            lambda_h = iszero(lambda_h) ? lambda : lambda_h
        end
        if obj == :div
            lambda_w = max(lambda_w, sqrt(eps(T)))
            lambda_h = max(lambda_h, sqrt(eps(T)))
        end
        new{T}(obj, maxiter, verbose, tol, update_H, lambda_w, lambda_h)
    end
end

function solve!(alg::MultUpdate{T}, X, W, H; io::IO=stdout, rng::AbstractRNG=default_rng()) where T

    Base.require_one_based_indexing(X, W, H)
    if alg.obj == :mse
        nmf_skeleton!(io, MultUpdMSE(alg.update_H, alg.lambda_w, alg.lambda_h, sqrt(eps(T))), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    else # alg == :div
        nmf_skeleton!(io, MultUpdDiv(alg.update_H, alg.lambda_w, alg.lambda_h, sqrt(eps(T))), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    end
end

# the multiplicative updating algorithm for MSE objective

struct MultUpdMSE{T} <: NMFUpdater{T}
    update_H::Bool
    lambda_w::T
    lambda_h::T
    delta::T
end

struct MultUpdMSE_State{T}
    WH::Matrix{T}
    WtX::Matrix{T}
    WtWH::Matrix{T}
    XHt::Matrix{T}
    WHHt::Matrix{T}

    function MultUpdMSE_State{T}(X, W::Matrix{T}, H::Matrix{T}) where T
        p, n, k = nmf_checksize(X, W, H)
        new{T}(W * H,
               Matrix{T}(undef, k, n),
               Matrix{T}(undef, k, n),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, p, k))
    end
end

prepare_state(::MultUpdMSE{T}, X, W, H) where {T} = MultUpdMSE_State{T}(X, W, H)
evaluate_objv(::MultUpdMSE{T}, s::MultUpdMSE_State{T}, X, W, H) where T = convert(T, 0.5) * sqL2dist(X, s.WH)

function update_wh!(upd::MultUpdMSE{T}, s::MultUpdMSE_State{T}, X, W::Matrix{T}, H::Matrix{T}) where T

    # fields
    lambda_w = upd.lambda_w
    lambda_h = upd.lambda_h
    delta = upd.delta
    WH = s.WH
    WtX = s.WtX
    WtWH = s.WtWH
    XHt = s.XHt
    WHHt = s.WHHt

    # update H
    if upd.update_H
        Wt = transpose(W)
        mul!(WtX, Wt, X)
        mul!(WtWH, Wt, WH)

        @inbounds for i = 1:length(H)
            H[i] *= (max(zero(T), WtX[i] - lambda_h) / (WtWH[i] + delta))
        end
        mul!(WH, W, H)
    end

    # update W
    Ht = transpose(H)
    mul!(XHt, X, Ht)
    mul!(WHHt, WH, Ht)

    @inbounds for i = 1:length(W)
        W[i] *= (max(zero(T), XHt[i] - lambda_w) / (WHHt[i] + delta))
    end
    mul!(WH, W, H)
end


# the multiplicative updating algorithm for divergence objective

struct MultUpdDiv{T} <: NMFUpdater{T}
    update_H::Bool
    lambda_w::T
    lambda_h::T
    delta::T
end

struct MultUpdDiv_State{T}
    WH::Matrix{T}
    sW::Matrix{T}     # sum(W, 1)
    sH::Matrix{T}     # sum(H, 2)
    Q::Matrix{T}      # X ./ (WH + lambda): size (p, n)
    WtQ::Matrix{T}    # W' * Q: size (k, n)
    QHt::Matrix{T}    # Q * H': size (p, k)

    function MultUpdDiv_State{T}(X, W::Matrix{T}, H::Matrix{T}) where T
        p, n, k = nmf_checksize(X, W, H)
        new{T}(W * H,
               Matrix{T}(undef, 1, k),
               Matrix{T}(undef, k, 1),
               Matrix{T}(undef, p, n),
               Matrix{T}(undef, k, n),
               Matrix{T}(undef, p, k))
    end
end

prepare_state(::MultUpdDiv{T}, X, W, H) where T = MultUpdDiv_State{T}(X, W, H)
evaluate_objv(::MultUpdDiv, s::MultUpdDiv_State, X, W, H) = gkldiv(X, s.WH)

function update_wh!(upd::MultUpdDiv{T}, s::MultUpdDiv_State{T}, X, W::Matrix{T}, H::Matrix{T}) where T

    p = size(X, 1)
    n = size(X, 2)
    k = size(W, 2)
    pn = p * n

    # fields
    lambda_w = upd.lambda_w
    lambda_h = upd.lambda_h
    delta = upd.delta
    sW = s.sW
    sH = s.sH
    WH = s.WH
    Q = s.Q
    WtQ = s.WtQ
    QHt = s.QHt

    @assert size(Q) == size(X)

    # update H
    if upd.update_H
        @inbounds for i = 1:length(X)
            Q[i] = X[i] / (WH[i] + delta)
        end
        mul!(WtQ, transpose(W), Q)
        sum!(fill!(sW, 0), W)
        @inbounds for j = 1:n, i = 1:k
            H[i,j] *= (WtQ[i,j] / (sW[i] + lambda_h))
        end
        mul!(WH, W, H)
    end

    # update W
    @inbounds for i = 1:length(X)
        Q[i] = X[i] / (WH[i] + delta)
    end
    mul!(QHt, Q, transpose(H))
    sum!(fill!(sH, 0), H)
    @inbounds for j = 1:k, i = 1:p
        W[i,j] *= (QHt[i,j] / (sH[j] + lambda_w))
    end
    mul!(WH, W, H)
end
