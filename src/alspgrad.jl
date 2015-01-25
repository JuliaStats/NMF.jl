# Alternate Least Squared by Projected Gradient Descent
#
#  Reference: Chih-Jen Lin. Projected Gradient Methods for Non-negative
#  Matrix Factorization. Neural Computing, 19 (2007).
#

## auxiliary routines

function projgradnorm(g, x)
    T = eltype(g)
    v = convert(T, 0.0)
    @inbounds for i = 1:length(g)
        gi = g[i]
        if gi < zero(T) || x[i] > zero(T)
            v += abs2(gi)
        end
    end
    return sqrt(v)
end

# Determine the maximum step size we can take without
# all elements of A becoming zero
# The new value of A would be A = A - α*G
function maxstep(G, A)
    T = typeof(one(eltype(A))/one(eltype(G)))
    αmax = zero(T)
    for i = 1:length(G)
        g = G[i]
        if g >= 0
            αmax = max(αmax, A[i]/g)
        else
            αmax = convert(T, Inf)
            break
        end
    end
    αmax
end

## sub-routines for updating H

immutable ALSGradUpdH_State{T}
    G::Matrix{T}      # gradient
    Hn::Matrix{T}     # newH in back-tracking
    Hp::Matrix{T}     # previous newH
    D::Matrix{T}      # Hn - H
    WtW::Matrix{T}    # W'W  (pre-computed)
    WtX::Matrix{T}    # W'X  (pre-computed)
    WtWD::Matrix{T}   # W'W * D

    function ALSGradUpdH_State(X, W, H)
        k, n = size(H)
        new(Array(T, k, n),
            Array(T, k, n),
            Array(T, k, n),
            Array(T, k, n),
            Array(T, k, k),
            Array(T, k, n),
            Array(T, k, n))
    end
end
ALSGradUpdH_State{T}(X, W::VecOrMat{T}, H::VecOrMat{T}) = ALSGradUpdH_State{T}(X, W, H)

function set_w!(s::ALSGradUpdH_State, X, W)
    At_mul_B!(s.WtW, W, W)
    At_mul_B!(s.WtX, W, X)
end

function alspgrad_updateh!{T}(X,
                              W::VecOrMat{T},
                              H::VecOrMat{T};
                              maxiter::Int = 1000,
                              traceiter::Int = 20,
                              tolg::T = cbrt(eps(T)),
                              beta::T = convert(T, 0.2),
                              sigma::T = convert(T, 0.01),
                              verbose::Bool = false)

    s = ALSGradUpdH_State(X, W, H)
    set_w!(s, X, W)
    _alspgrad_updateh!(X, W, H, s, 
                       maxiter, traceiter, tolg, 
                       beta, sigma, verbose)
end

function _alspgrad_updateh!(X,                      # size (p, n)
                            W::VecOrMat,            # size (p, k)
                            H::VecOrMat,            # size (k, n)
                            s::ALSGradUpdH_State,   # state to hold temporary quantities
                            maxiter::Int,           # the maximum number of (outer) iterations
                            traceiter::Int,         # the number of iterations to trace alpha
                            tolg,                   # first-order optimality tolerance
                            β,                      # the value of beta (back-tracking ratio)
                            σ,                      # the value of sigma
                            verbose::Bool)          # whether to show procedural info
    # fields
    G = s.G
    Hn = s.Hn
    Hp = s.Hp
    D = s.D
    WtW = s.WtW
    WtX = s.WtX
    WtWD = s.WtWD
    T = eltype(H)

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
    α = one(T)/one(eltype(G))
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

        αmax = maxstep(G, H)
        α = min(α, (β+3eps(β))*αmax)

        # back-tracking
        it = 0
        if !converged
            while it < traceiter
                it += 1
                isfinite(α) || error("α is not finite")

                # projected update of H to Hn
                @inbounds for i = 1:length(Hn)
                    hi = H[i]
                    Hn[i] = hi_ = max(hi - α * G[i], zero(T))
                    D[i] = hi_ - hi
                end

                # compute criterion
                dv1 = BLAS.dot(G, D)  # <G, D>
                A_mul_B!(WtWD, WtW, D)
                dv2 = BLAS.dot(WtWD, D)  # <D, WtW * D>
                
                # back-track
                suff_decr = ((1 - σ) * dv1 + convert(T, 0.5) * dv2) < 0

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
                        if α/β < αmax
                            copy!(Hp, Hn)
                            α /= β
                        else
                            copy!(H, Hn)
                            break
                        end
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

immutable ALSGradUpdW_State{T}
    G::Matrix{T}      # gradient
    Wn::Matrix{T}     # newW in back-tracking
    Wp::Matrix{T}     # previous newW
    D::Matrix{T}      # Wn - W
    HHt::Matrix{T}    # HH' (pre-computed)
    XHt::Matrix{T}    # XH' (pre-computed)
    DHHt::Matrix{T}   # D * HH'

    function ALSGradUpdW_State(X, W, H)
        p, k = size(W)
        new(Array(T, p, k),
            Array(T, p, k),
            Array(T, p, k),
            Array(T, p, k),
            Array(T, k, k),
            Array(T, p, k),
            Array(T, p, k))
    end
end
ALSGradUpdW_State{T}(X, W::VecOrMat{T}, H::VecOrMat{T}) = ALSGradUpdW_State{T}(X, W, H)

function set_h!(s::ALSGradUpdW_State, X, H)
    A_mul_Bt!(s.HHt, H, H)
    A_mul_Bt!(s.XHt, X, H)
end


function alspgrad_updatew!{T}(X,
                              W::VecOrMat{T},
                              H::VecOrMat{T};
                              maxiter::Int = 1000,
                              traceiter::Int = 20,
                              tolg::T = cbrt(eps(T)),
                              beta::T = convert(T, 0.2),
                              sigma::T = convert(T, 0.01),
                              verbose::Bool = false)

    s = ALSGradUpdW_State(X, W, H)
    set_h!(s, X, H)
    _alspgrad_updatew!(X, W, H, s, 
                       maxiter, traceiter, tolg, 
                       beta, sigma, verbose)
end

function _alspgrad_updatew!(X,                      # size (p, n)
                            W::VecOrMat,            # size (p, k)
                            H::VecOrMat,            # size (k, n)
                            s::ALSGradUpdW_State,   # state to hold temporary quantities
                            maxiter::Int,           # the maximum number of (outer) iterations
                            traceiter::Int,         # the number of iterations to trace alpha
                            tolg,                   # first-order optimality tolerance
                            β,                      # the value of beta (back-tracking ratio)
                            σ,                      # the value of sigma
                            verbose::Bool)          # whether to show procedural info
    # fields
    G = s.G
    Wn = s.Wn
    Wp = s.Wp
    D = s.D
    HHt = s.HHt
    XHt = s.XHt
    DHHt = s.DHHt
    T = eltype(W)

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
    α = one(T)/one(eltype(G))
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

        αmax = maxstep(G, W)
        α = min(α, (β+3eps(β))*αmax)

        # back-tracking
        it = 0
        if !converged
            while it < traceiter
                it += 1
                isfinite(α) || error("α is not finite")

                # projected update of W to Wn
                @inbounds for i = 1:length(Wn)
                    wi = W[i]
                    Wn[i] = wi_ = max(wi - α * G[i], zero(T))
                    D[i] = wi_ - wi
                end

                # compute criterion
                dv1 = BLAS.dot(G, D)  # <G, D>
                A_mul_B!(DHHt, D, HHt)
                dv2 = BLAS.dot(DHHt, D)  # <D * HHt, D>
                
                # back-track
                suff_decr = ((1 - σ) * dv1 + convert(T, 0.5) * dv2) < 0

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
                        if α/β < αmax
                            copy!(Wp, Wn)
                            α /= β
                        else
                            copy!(W, Wn)
                            break
                        end
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

type ALSPGrad{T}
    maxiter::Int      # maximum number of main iterations
    maxsubiter::Int   # maximum number of iterations within a sub-routine
    tol::T      # tolerance of changes on W & H (main)
    tolg::T     # tolerance of grad norm in sub-routine
    verbose::Bool     # whether to show procedural information (main)

    function ALSPGrad(;maxiter::Integer=100, 
                       maxsubiter::Integer=200,
                       tol::Real=cbrt(eps(T)),
                       tolg::Real=eps(T)^(1/4),
                       verbose::Bool=false)
        new(maxiter,
            maxsubiter,
            tol,
            tolg,
            verbose)
    end
end

immutable ALSPGradUpd{T} <: NMFUpdater{T}
    maxsubiter::Int
    tolg::T
end

solve!(alg::ALSPGrad, X, W, H) =
    nmf_skeleton!(ALSPGradUpd(alg.maxsubiter, alg.tolg), 
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


immutable ALSPGradUpd_State{T}
    WH::Matrix{T}
    uhstate::ALSGradUpdH_State
    uwstate::ALSGradUpdW_State

    ALSPGradUpd_State(X, W, H) =
        new(W * H, 
            ALSGradUpdH_State(X, W, H), 
            ALSGradUpdW_State(X, W, H))
end

prepare_state{T}(::ALSPGradUpd{T}, X, W, H) = ALSPGradUpd_State{T}(X, W, H)
evaluate_objv(u::ALSPGradUpd, s::ALSPGradUpd_State, X, W, H) = sqL2dist(X, s.WH)

function update_wh!(upd::ALSPGradUpd, s::ALSPGradUpd_State, X, W, H)
    T = eltype(W)

    # update H
    set_w!(s.uhstate, X, W)
    _alspgrad_updateh!(X, W, H, s.uhstate, 
        upd.maxsubiter, 20, upd.tolg, convert(T, 0.2), convert(T, 0.01), false)

    # update W
    set_h!(s.uwstate, X, H)
    _alspgrad_updatew!(X, W, H, s.uwstate, 
        upd.maxsubiter, 20, upd.tolg, convert(T, 0.2), convert(T, 0.01), false)

    # update WH
    A_mul_B!(s.WH, W, H) 
end

