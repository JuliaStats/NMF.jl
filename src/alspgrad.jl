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

## sub-routines for updating H

immutable ALSGradUpdH_State
    G::Matrix{Float64}      # gradient
    Hn::Matrix{Float64}     # newH in back-tracking
    Hp::Matrix{Float64}     # previous newH
    D::Matrix{Float64}      # Hn - H
    WtW::Matrix{Float64}    # W'W  (pre-computed)
    WtX::Matrix{Float64}    # W'X  (pre-computed)
    WtWD::Matrix{Float64}   # W'W * D

    function ALSGradUpdH_State(X::ContiguousMatrix, W::ContiguousMatrix, H::ContiguousMatrix)
        k, n = size(H)
        new(Array(Float64, k, n), 
            Array(Float64, k, n),
            Array(Float64, k, n), 
            Array(Float64, k, n),
            Array(Float64, k, k),
            Array(Float64, k, n), 
            Array(Float64, k, n))
    end
end

function set_w!(s::ALSGradUpdH_State, X::ContiguousMatrix, W::ContiguousMatrix)
    At_mul_B!(s.WtW, W, W)
    At_mul_B!(s.WtX, W, X)
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
    set_w!(s, X, W)
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

                # projected update of H to Hn
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


## sub-routines for updating W

immutable ALSGradUpdW_State
    G::Matrix{Float64}      # gradient
    Wn::Matrix{Float64}     # newW in back-tracking
    Wp::Matrix{Float64}     # previous newW
    D::Matrix{Float64}      # Wn - W
    HHt::Matrix{Float64}    # HH' (pre-computed)
    XHt::Matrix{Float64}    # XH' (pre-computed)
    DHHt::Matrix{Float64}   # D * HH'

    function ALSGradUpdW_State(X::ContiguousMatrix, W::ContiguousMatrix, H::ContiguousMatrix)
        p, k = size(W)
        new(Array(Float64, p, k), 
            Array(Float64, p, k),
            Array(Float64, p, k), 
            Array(Float64, p, k),
            Array(Float64, k, k),
            Array(Float64, p, k), 
            Array(Float64, p, k))
    end
end

function set_h!(s::ALSGradUpdW_State, X::ContiguousMatrix, H::ContiguousMatrix)
    A_mul_Bt!(s.HHt, H, H)
    A_mul_Bt!(s.XHt, X, H)
end


function alspgrad_updatew!(X::Matrix{Float64}, 
                           W::Matrix{Float64}, 
                           H::Matrix{Float64};
                           maxiter::Int = 1000, 
                           traceiter::Int = 20,
                           tolg::Float64 = 1.0e-6,
                           beta::Float64 = 0.2, 
                           sigma::Float64 = 0.01, 
                           verbose::Bool = false)

    s = ALSGradUpdW_State(X, W, H)
    set_h!(s, X, H)
    _alspgrad_updatew!(X, W, H, s, 
                       maxiter, traceiter, tolg, 
                       beta, sigma, verbose)
end

function _alspgrad_updatew!(X::Matrix{Float64},     # size (p, n)
                            W::Matrix{Float64},     # size (p, k)
                            H::Matrix{Float64},     # size (k, n)
                            s::ALSGradUpdW_State,   # state to hold temporary quantities
                            maxiter::Int,           # the maximum number of (outer) iterations
                            traceiter::Int,         # the number of iterations to trace alpha
                            tolg::Float64,          # first-order optimality tolerance
                            β::Float64,             # the value of beta (back-tracking ratio)
                            σ::Float64,             # the value of sigma                            
                            verbose::Bool)          # whether to show procedural info
    # fields
    G::Matrix{Float64} = s.G
    Wn::Matrix{Float64} = s.Wn
    Wp::Matrix{Float64} = s.Wp
    D::Matrix{Float64} = s.D
    HHt::Matrix{Float64} = s.HHt
    XHt::Matrix{Float64} = s.XHt
    DHHt::Matrix{Float64} = s.DHHt

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
        A_mul_B!(G, W, HHt)
        for i = 1:length(G)
            G[i] -= XHt[i]
        end

        # compute projected norm of gradient
        pgnrm = projgradnorm(G, W)
        if pgnrm < tolg
            converged = true
        end

        # back-tracking
        it = 0
        if !converged
            while it < traceiter
                it += 1

                # projected update of W to Wn
                @inbounds for i = 1:length(Wn)
                    wi = W[i]
                    Wn[i] = wi_ = max(wi - α * G[i], 0.0)
                    D[i] = wi_ - wi
                end

                # compute criterion
                dv1 = BLAS.dot(G, D)  # <G, D>
                A_mul_B!(DHHt, D, HHt)
                dv2 = BLAS.dot(DHHt, D)  # <D * HHt, D>
                
                # back-track
                suff_decr = ((1 - σ) * dv1 + 0.5 * dv2) < 0.0

                if it == 1
                    to_decr = !suff_decr
                end

                if to_decr
                    if !suff_decr
                        α *= β
                    else
                        copy!(W, Wn)
                        break
                    end
                else
                    if suff_decr
                        copy!(Wp, Wn)
                        α /= β
                    else
                        copy!(W, Wp)
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


## main algorithm

type ALSPGrad
    maxiter::Int      # maximum number of main iterations
    maxsubiter::Int   # maximum number of iterations within a sub-routine
    tol::Float64      # tolerance of changes on W & H (main)
    tolg::Float64     # tolerance of grad norm in sub-routine
    verbose::Bool     # whether to show procedural information (main)

    function ALSPGrad(;maxiter::Integer=100, 
                       maxsubiter::Integer=200,
                       tol::Real=1.0e-6, 
                       tolg::Real=1.0e-4, 
                       verbose::Bool=false)
        new(int(maxiter), 
            int(maxsubiter),
            float64(tol), 
            float64(tolg), 
            verbose)
    end
end

immutable ALSPGradUpd <: NMFUpdater
    maxsubiter::Int
    tolg::Float64  
end

solve!(alg::ALSPGrad, X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64}) =
    nmf_skeleton!(ALSPGradUpd(alg.maxsubiter, alg.tolg), 
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


immutable ALSPGradUpd_State
    WH::Matrix{Float64}
    uhstate::ALSGradUpdH_State
    uwstate::ALSGradUpdW_State

    ALSPGradUpd_State(X::ContiguousMatrix, W::ContiguousMatrix, H::ContiguousMatrix) = 
        new(W * H, 
            ALSGradUpdH_State(X, W, H), 
            ALSGradUpdW_State(X, W, H))
end

prepare_state(::ALSPGradUpd, X, W, H) = ALSPGradUpd_State(X, W, H)
evaluate_objv(u::ALSPGradUpd, s::ALSPGradUpd_State, X, W, H) = sqL2dist(X, s.WH)

function update_wh!(upd::ALSPGradUpd, s::ALSPGradUpd_State, 
                    X::Matrix{Float64}, 
                    W::Matrix{Float64}, 
                    H::Matrix{Float64})

    # update H
    set_w!(s.uhstate, X, W)
    _alspgrad_updateh!(X, W, H, s.uhstate, 
        upd.maxsubiter, 20, upd.tolg, 0.2, 0.01, false)

    # update W
    set_h!(s.uwstate, X, H)
    _alspgrad_updatew!(X, W, H, s.uwstate, 
        upd.maxsubiter, 20, upd.tolg, 0.2, 0.01, false)

    # update WH
    A_mul_B!(s.WH, W, H) 
end


