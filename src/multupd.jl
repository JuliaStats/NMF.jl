#
# Multiplicative updating algorithm
#
# Reference: 
#   Daniel D. Lee and H. Sebastian Seung. Algorithms for Non-negative 
#   Matrix Factorization. Advances in NIPS, 2001.
#

type MultUpdate
    obj::Symbol         # objective :mse or :div
    maxiter::Int        # maximum number of iterations
    verbose::Bool       # whether to show procedural information
    tol::Float64        # change tolerance upon convergence
    lambda::Float64     # regularization coefficient

    function MultUpdate(;obj::Symbol=:mse,
                         maxiter::Integer=100, 
                         verbose::Bool=false,
                         tol::Real=1.0e-6, 
                         lambda::Real=1.0e-9)

        obj == :mse || obj == :div || error("Invalid value for obj.")
        maxiter > 1 || error("maxiter must be greater than 1.")
        tol > 0 || error("tol must be positive.")
        lambda >= 0 || error("lambda must be non-negative.")

        new(obj, 
            int(maxiter), 
            verbose, 
            float64(tol), 
            float64(lambda))
    end
end

function nmf_solve!(alg::MultUpdate, 
                    X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64})

    if alg.obj == :mse
        nmf_skeleton!(MultUpdMSE(alg.lambda), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    else # alg == :div
        nmf_skeleton!(MultUpdDiv(alg.lambda), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    end
end

# the multiplicative updating algorithm for MSE objective

immutable MultUpdMSE <: NMFUpdater 
    lambda::Float64
end

immutable MultUpdMSE_State
    WH::Matrix{Float64}
    WtX::Matrix{Float64}
    WtWH::Matrix{Float64}
    XHt::Matrix{Float64}
    WHHt::Matrix{Float64}

    function MultUpdMSE_State(X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64})
        p, n, k = nmf_checksize(X, W, H)
        new(W * H, 
            Array(Float64, k, n), 
            Array(Float64, k, n), 
            Array(Float64, p, k), 
            Array(Float64, p, k))
    end
end

prepare_state(::MultUpdMSE, X, W, H) = MultUpdMSE_State(X, W, H)
evaluate_objv(::MultUpdMSE, s::MultUpdMSE_State, X, W, H) = msd(X, s.WH)

function update_wh!(upd::MultUpdMSE, s::MultUpdMSE_State, 
                    X::Matrix{Float64}, 
                    W::Matrix{Float64}, 
                    H::Matrix{Float64})

    # fields
    lambda::Float64 = upd.lambda
    WH::Matrix{Float64} = s.WH
    WtX::Matrix{Float64} = s.WtX
    WtWH::Matrix{Float64} = s.WtWH
    XHt::Matrix{Float64} = s.XHt
    WHHt::Matrix{Float64} = s.WHHt

    # update H
    At_mul_B!(WtX, W, X)
    At_mul_B!(WtWH, W, WH)

    for i = 1:length(H)
        H[i] *= (WtX[i] / (WtWH[i] + lambda))
    end
    A_mul_B!(WH, W, H)

    # update W
    A_mul_Bt!(XHt, X, H)
    A_mul_Bt!(WHHt, WH, H)

    for i = 1:length(W)
        W[i] *= (XHt[i] / (WHHt[i] + lambda))
    end
    A_mul_B!(WH, W, H)
end


# the multiplicative updating algorithm for divergence objective

immutable MultUpdDiv <: NMFUpdater 
    lambda::Float64
end

immutable MultUpdDiv_State
    WH::Matrix{Float64}     
    sW::Matrix{Float64}     # sum(W, 1)
    sH::Matrix{Float64}     # sum(H, 2)
    Q::Matrix{Float64}      # X ./ (WH + lambda): size (p, n)
    WtQ::Matrix{Float64}    # W' * Q: size (k, n)
    QHt::Matrix{Float64}    # Q * H': size (p, k)

    function MultUpdDiv_State(X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64})
        p, n, k = nmf_checksize(X, W, H)
        new(W * H, 
            Array(Float64, 1, k),
            Array(Float64, k, 1),
            Array(Float64, p, n), 
            Array(Float64, k, n), 
            Array(Float64, p, k))
    end
end

prepare_state(::MultUpdDiv, X, W, H) = MultUpdDiv_State(X, W, H)
evaluate_objv(::MultUpdDiv, s::MultUpdDiv_State, X, W, H) = gkldiv(X, s.WH)

function update_wh!(upd::MultUpdDiv, s::MultUpdDiv_State, 
                    X::Matrix{Float64}, 
                    W::Matrix{Float64}, 
                    H::Matrix{Float64})

    p = size(X, 1)
    n = size(X, 2)
    k = size(W, 2)
    pn = p * n

    # fields
    lambda::Float64 = upd.lambda
    sW::Matrix{Float64} = s.sW
    sH::Matrix{Float64} = s.sH
    WH::Matrix{Float64} = s.WH
    Q::Matrix{Float64} = s.Q
    WtQ::Matrix{Float64} = s.WtQ
    QHt::Matrix{Float64} = s.QHt

    @assert size(Q) == size(X)

    # update H
    for i = 1:length(X)
        Q[i] = X[i] / (WH[i] + lambda)
    end
    At_mul_B!(WtQ, W, Q)
    sum!(fill!(sW, 0.0), W)
    for j = 1:n, i = 1:k
        H[i,j] *= (WtQ[i,j] / sW[i])
    end
    A_mul_B!(WH, W, H)

    # update W
    for i = 1:length(X)
        Q[i] = X[i] / (WH[i] + lambda)
    end
    A_mul_Bt!(QHt, Q, H)
    sum!(fill!(sH, 0.0), H)
    for j = 1:k, i = 1:p
        W[i,j] *= (QHt[i,j] / sH[j])
    end
    A_mul_B!(WH, W, H)
end

