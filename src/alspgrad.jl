# Alternate Least Squared by Projected Gradient Descent
#
#  Reference: Chih-Jen Lin. Projected Gradient Methods for Non-negative
#  Matrix Factorization. Neural Computing, 19 (2007).
#

## auxiliary routines

function projgradnorm(g::ContiguousArray{Float64}, x::ContiguousArray{Float64})
    v = 0.0
    @inbounds for i = 1:length(g)
        gi = g[i]
        if gi < 0.0 || x[i] > 0.0
            v += abs2(gi)
        end
    end
    return sqrt(v)
end


immutable ALSGradUpdH_State
    G::Matrix{Float64}      # gradient
    Htmp::Matrix{Float64}   # temporary H in back-tracking
    D::Matrix{Float64}      # Htmp - H
    WtW::Matrix{Float64}    # W'W  (pre-computed)
    WtX::Matrix{Float64}    # W'X  (pre-computed)
    WtWD::Matrix{Float64}   # W'W * D

    function ALSGradUpdH_State(X::ContiguousMatrix, W::ContiguousMatrix, H::ContiguousMatrix)
        k, n = size(H)
        new(Array(Float64, k, n), 
            Array(Float64, k, n),
            Array(Float64, k, n), 
            At_mul_B(W, W),
            At_mul_B(W, X), 
            Array(Float64, k, n))
    end
end

function _alspgrad_updateh!(X::Matrix{Float64},     # size (p, n)
                            W::Matrix{Float64},     # size (p, k)
                            H::Matrix{Float64},     # size (k, n)
                            s::ALSGradUpdH_State,   # state to hold temporary quantities
                            maxiter::Int,       # the maximum number of (outer) iterations
                            traceiter::Int,     # the number of iterations to trace alpha
                            tolg::Float64,      # first-order optimality tolerance
                            α::Float64,         # the initial value of alpha
                            β::Float64,         # the value of beta (back-tracking ratio)
                            σ::Float64,         # the value of sigma                            
                            verbose::Bool)      # whether to show procedural info
    # fields
    G::Matrix{Float64} = s.G
    D::Matrix{Float64} = s.D
    WtW::Matrix{Float64} = s.WtW
    WtX::Matrix{Float64} = s.WtX
    WtWD::Matrix{Float64} = s.WtWD

    t = 0
    converged = false
    while t < maxiter
        t += 1

        # compute gradient
        A_mul_B!(G, WtW, H)
        for i = 1:length(G)
            G[i] -= WtX[i]
        end

        # compute projected norm of gradient
        pgnrm = projgradnorm(G, H)
        if pgnrm < tolg
            converged = true
            break
        end
        

    end
end


