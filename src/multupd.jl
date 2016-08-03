#
# Multiplicative updating algorithm
#
# Reference:
#   Daniel D. Lee and H. Sebastian Seung. Algorithms for Non-negative
#   Matrix Factorization. Advances in NIPS, 2001.
#

type MultUpdate{T}
    obj::Symbol         # objective :mse or :div
    maxiter::Int        # maximum number of iterations
    verbose::Bool       # whether to show procedural information
    tol::T              # change tolerance upon convergence
    lambda::T           # regularization coefficient

    function MultUpdate(;obj::Symbol=:mse,
                         maxiter::Integer=100,
                         verbose::Bool=false,
                         tol::Real=cbrt(eps(T)),
                         lambda::Real=sqrt(eps(T)))

        obj == :mse || obj == :div || throw(ArgumentError("Invalid value for obj."))
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        lambda >= 0 || throw(ArgumentError("lambda must be non-negative."))

        new(obj, maxiter, verbose, tol, lambda)
    end
end

function solve!(alg::MultUpdate, X, W, H)

    if alg.obj == :mse
        nmf_skeleton!(MultUpdMSE(alg.lambda), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    else # alg == :div
        nmf_skeleton!(MultUpdDiv(alg.lambda), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    end
end

# the multiplicative updating algorithm for MSE objective

immutable MultUpdMSE{T} <: NMFUpdater{T}
    lambda::T
end

immutable MultUpdMSE_State{T}
    WH::Matrix{T}
    WtX::Matrix{T}
    WtWH::Matrix{T}
    XHt::Matrix{T}
    WHHt::Matrix{T}

    function MultUpdMSE_State(X, W::Matrix{T}, H::Matrix{T})
        p, n, k = nmf_checksize(X, W, H)
        @compat new(W * H,
                    Array{T,2}(k, n),
                    Array{T,2}(k, n),
                    Array{T,2}(p, k),
                    Array{T,2}(p, k))
    end
end

prepare_state{T}(::MultUpdMSE{T}, X, W, H) = MultUpdMSE_State{T}(X, W, H)
evaluate_objv(::MultUpdMSE, s::MultUpdMSE_State, X, W, H) = sqL2dist(X, s.WH)

function update_wh!{T}(upd::MultUpdMSE{T}, s::MultUpdMSE_State{T}, X, W::Matrix{T}, H::Matrix{T})

    # fields
    lambda = upd.lambda
    WH = s.WH
    WtX = s.WtX
    WtWH = s.WtWH
    XHt = s.XHt
    WHHt = s.WHHt

    # update H
    At_mul_B!(WtX, W, X)
    At_mul_B!(WtWH, W, WH)

    @inbounds for i = 1:length(H)
        H[i] *= (WtX[i] / (WtWH[i] + lambda))
    end
    A_mul_B!(WH, W, H)

    # update W
    A_mul_Bt!(XHt, X, H)
    A_mul_Bt!(WHHt, WH, H)

    @inbounds for i = 1:length(W)
        W[i] *= (XHt[i] / (WHHt[i] + lambda))
    end
    A_mul_B!(WH, W, H)
end


# the multiplicative updating algorithm for divergence objective

immutable MultUpdDiv{T} <: NMFUpdater{T}
    lambda::T
end

immutable MultUpdDiv_State{T}
    WH::Matrix{T}
    sW::Matrix{T}     # sum(W, 1)
    sH::Matrix{T}     # sum(H, 2)
    Q::Matrix{T}      # X ./ (WH + lambda): size (p, n)
    WtQ::Matrix{T}    # W' * Q: size (k, n)
    QHt::Matrix{T}    # Q * H': size (p, k)

    function MultUpdDiv_State(X, W::Matrix{T}, H::Matrix{T})
        p, n, k = nmf_checksize(X, W, H)
        @compat new(W * H,
                    Array{T,2}(1, k),
                    Array{T,2}(k, 1),
                    Array{T,2}(p, n),
                    Array{T,2}(k, n),
                    Array{T,2}(p, k))
    end
end

prepare_state{T}(::MultUpdDiv{T}, X, W, H) = MultUpdDiv_State{T}(X, W, H)
evaluate_objv(::MultUpdDiv, s::MultUpdDiv_State, X, W, H) = gkldiv(X, s.WH)

function update_wh!{T}(upd::MultUpdDiv{T}, s::MultUpdDiv_State{T}, X, W::Matrix{T}, H::Matrix{T})

    p = size(X, 1)
    n = size(X, 2)
    k = size(W, 2)
    pn = p * n

    # fields
    lambda = upd.lambda
    sW = s.sW
    sH = s.sH
    WH = s.WH
    Q = s.Q
    WtQ = s.WtQ
    QHt = s.QHt

    @assert size(Q) == size(X)

    # update H
    @inbounds for i = 1:length(X)
        Q[i] = X[i] / (WH[i] + lambda)
    end
    At_mul_B!(WtQ, W, Q)
    sum!(fill!(sW, 0), W)
    @inbounds for j = 1:n, i = 1:k
        H[i,j] *= (WtQ[i,j] / sW[i])
    end
    A_mul_B!(WH, W, H)

    # update W
    @inbounds for i = 1:length(X)
        Q[i] = X[i] / (WH[i] + lambda)
    end
    A_mul_Bt!(QHt, Q, H)
    sum!(fill!(sH, 0), H)
    @inbounds for j = 1:k, i = 1:p
        W[i,j] *= (QHt[i,j] / sH[j])
    end
    A_mul_B!(WH, W, H)
end
