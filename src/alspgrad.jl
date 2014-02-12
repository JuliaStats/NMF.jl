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
    Hn::Matrix{Float64}     # newH in back-tracking
    Hp::Matrix{Float64}     # previous newH
    D::Matrix{Float64}      # Htmp - H
    WtW::Matrix{Float64}    # W'W  (pre-computed)
    WtX::Matrix{Float64}    # W'X  (pre-computed)
    WtWD::Matrix{Float64}   # W'W * D

    function ALSGradUpdH_State(X::ContiguousMatrix, W::ContiguousMatrix, H::ContiguousMatrix)
        k, n = size(H)
        new(Array(Float64, k, n), 
            Array(Float64, k, n),
            Array(Float64, k, n), 
            Array(Float64, k, n),
            At_mul_B(W, W),
            At_mul_B(W, X), 
            Array(Float64, k, n))
    end
end

function alspgrad_updateh!(X::Matrix{Float64}, 
                           W::Matrix{Float64}, 
                           H::Matrix{Float64};
                           maxiter::Int = 1000, 
                           traceiter::Int = 20,
                           tolg::Float64 = 1.0e-6,
                           beta::Float64 = 0.2, 
                           sigma::Float64 = 0.01, 
                           verbose::Bool = false)

    s = ALSGradUpdH_State(X, W, H)
    _alspgrad_updateh!(X, W, H, s, 
                       maxiter, traceiter, tolg, 
                       beta, sigma, verbose)
end


function _alspgrad_updateh!(X::Matrix{Float64},     # size (p, n)
                            W::Matrix{Float64},     # size (p, k)
                            H::Matrix{Float64},     # size (k, n)
                            s::ALSGradUpdH_State,   # state to hold temporary quantities
                            maxiter::Int,           # the maximum number of (outer) iterations
                            traceiter::Int,         # the number of iterations to trace alpha
                            tolg::Float64,          # first-order optimality tolerance
                            β::Float64,             # the value of beta (back-tracking ratio)
                            σ::Float64,             # the value of sigma                            
                            verbose::Bool)          # whether to show procedural info
    # fields
    G::Matrix{Float64} = s.G
    Hn::Matrix{Float64} = s.Hn
    Hp::Matrix{Float64} = s.Hp
    D::Matrix{Float64} = s.D
    WtW::Matrix{Float64} = s.WtW
    WtX::Matrix{Float64} = s.WtX
    WtWD::Matrix{Float64} = s.WtWD

    # banner
    if verbose       
        @printf("%5s    %12s    %12s    %12s    %8s    %12s\n", 
            "Iter", "objv", "objv.change", "1st-ord", "alpha", "back-tracks")
        WH = W * H
        objv = sqL2dist(X, WH)
        @printf("%5d    %12.5e\n", 0, objv)
    end

    # main loop
    t = 0
    converged = false
    to_decr = true
    α = 1.0
    while !converged && t < maxiter
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
        end

        # back-tracking
        it = 0
        if !converged
            while it < traceiter
                it += 1

                # projected update of H to H_
                @inbounds for i = 1:length(Hn)
                    hi = H[i]
                    Hn[i] = hi_ = max(hi - α * G[i], 0.0)
                    D[i] = hi_ - hi
                end

                # compute criterion
                dv1 = BLAS.dot(G, D)  # <G, D>
                A_mul_B!(WtWD, WtW, D)
                dv2 = BLAS.dot(WtWD, D)  # <D, WtW * D>
                
                # back-track
                suff_decr = ((1 - σ) * dv1 + 0.5 * dv2) < 0.0

                if it == 1
                    to_decr = !suff_decr
                end

                if to_decr
                    if !suff_decr
                        α *= β
                    else
                        copy!(H, Hn)
                        break
                    end
                else
                    if suff_decr
                        copy!(Hp, Hn)
                        α /= β
                    else
                        copy!(H, Hp)
                        break
                    end
                end
            end
        end

        # print info
        if verbose
            A_mul_B!(WH, W, H)
            preobjv = objv
            objv = sqL2dist(X, WH)
            @printf("%5d    %12.5e    %12.5e    %12.5e    %8.4f    %12d\n", 
                t, objv, objv - preobjv, pgnrm, α, it)
        end
    end
    return H
end


