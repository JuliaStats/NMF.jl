#
# Multiplicative updating algorithm
#
# Reference: 
#   Daniel D. Lee and H. Sebastian Seung. Algorithms for Non-negative 
#   Matrix Factorization. Advances in NIPS, 2001.
#

type NMFMultUpdate
    obj::Symbol         # objective :mse or :div
    maxiter::Int        # maximum number of iterations
    verbose::Bool       # whether to show procedural information
    tol::Float64        # change tolerance upon convergence
    lambda::Float64     # regularization coefficient

    function NMFMultUpdate(;obj::Symbol=:mse,
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

function nmf_solve!(alg::NMFMultUpdate, 
                    X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64})

    if alg.obj == :mse
        nmf_skeleton!(NMFMultUpdMSE(alg.lambda), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    else # alg == :div
        nmf_skeleton!(NMFMultUpdDiv(alg.lambda), X, W, H, alg.maxiter, alg.verbose, alg.tol)
    end
end

# the multiplicative updating algorithm for MSE objective

immutable NMFMultUpdMSE <: NMFUpdater 
    lambda::Float64
end

immutable NMFMultUpdMSE_State
    WH::Matrix{Float64}
    WtX::Matrix{Float64}
    WtWH::Matrix{Float64}
    XHt::Matrix{Float64}
    WHHt::Matrix{Float64}

    function NMFMultUpdMSE_State(X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64})
        p, n, k = nmf_checksize(X, W, H)
        new(W * H, 
            Array(Float64, k, n), 
            Array(Float64, k, n), 
            Array(Float64, p, k), 
            Array(Float64, p, k))
    end
end

prepare_state(::NMFMultUpdMSE, X, W, H) = NMFMultUpdMSE_State(X, W, H)
evaluate_objv(::NMFMultUpdMSE, s::NMFMultUpdMSE_State, X, W, H) = msd(X, s.WH)

function update_wh!(upd::NMFMultUpdMSE, s::NMFMultUpdMSE_State, 
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

