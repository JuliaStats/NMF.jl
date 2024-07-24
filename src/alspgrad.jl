# Alternate Least Squared by Projected Gradient Descent
#
#  Reference: Chih-Jen Lin. Projected Gradient Methods for Non-negative
#  Matrix Factorization. Neural Computing, 19 (2007).
#

## auxiliary routines

function projgradnorm(g, x)
    T = eltype(g)
    v = zero(T)
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
    T = typeof(one(eltype(A)) / one(eltype(G)))
    αmax = zero(T)
    for i = 1:length(G)
        g = G[i]
        if g >= 0
            αmax = max(αmax, A[i] / g)
        else
            αmax = convert(T, Inf)
            break
        end
    end
    return αmax
end

## sub-routines for updating H

struct ALSGradUpdH_State{T}
    G::Matrix{T}      # gradient
    Hn::Matrix{T}     # newH in back-tracking
    Hp::Matrix{T}     # previous newH
    D::Matrix{T}      # Hn - H
    WtW::Matrix{T}    # W'W  (pre-computed)
    WtX::Matrix{T}    # W'X  (pre-computed)
    WtWD::Matrix{T}   # W'W * D

    function ALSGradUpdH_State{T}(X, W, H) where T
        k, n = size(H)
        new{T}(Matrix{T}(undef, k, n),
               Matrix{T}(undef, k, n),
               Matrix{T}(undef, k, n),
               Matrix{T}(undef, k, n),
               Matrix{T}(undef, k, k),
               Matrix{T}(undef, k, n),
               Matrix{T}(undef, k, n))
    end
end
ALSGradUpdH_State(X, W::VecOrMat{T}, H::VecOrMat{T}) where {T} = ALSGradUpdH_State{T}(X, W, H)

function set_w!(s::ALSGradUpdH_State, X, W)
    Wt = transpose(W)
    mul!(s.WtW, Wt, W)
    mul!(s.WtX, Wt, X)
end

function alspgrad_updateh!(X,
                           W::VecOrMat{T},
                           H::VecOrMat{T};
                           maxiter::Int = 1000,
                           traceiter::Int = 20,
                           tolg::T = cbrt(eps(T)),
                           beta::T = convert(T, 0.2),
                           sigma::T = convert(T, 0.01),
                           verbose::Bool = false) where T

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
        objv = convert(T, 0.5) * sqL2dist(X, WH)
        @printf("%5d    %12.5e\n", 0, objv)
    end

    # main loop
    t = 0
    converged = false
    decr_alpha = true
    α = one(T) / one(eltype(G))
    while !converged && t < maxiter
        t += 1

        # compute gradient
        mul!(G, WtW, H)
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
                isfinite(α) || error("α is not finite")

                # projected update of H to Hn
                @inbounds for i = 1:length(Hn)
                    hi = H[i]
                    Hn[i] = hi_ = max(hi - α * G[i], zero(T))
                    D[i] = hi_ - hi
                end

                # compute criterion
                dv1 = BLAS.dot(G, D)  # <G, D>
                mul!(WtWD, WtW, D)
                dv2 = BLAS.dot(WtWD, D)  # <D, WtW * D>

                # back-track
                suff_decr = ((1 - σ) * dv1 + convert(T, 0.5) * dv2) < 0

                if it == 1
                    decr_alpha = !suff_decr
                    copyto!(Hp, H)
                end

                if decr_alpha
                    if suff_decr
                        copyto!(H, Hn)
                        break
                    else
                        α *= β
                    end
                else
                    if !suff_decr || isapprox(Hp, Hn, atol=eps(T))
                        copyto!(H, Hp)
                        break
                    else
                        α /= β
                        copyto!(Hp, Hn)
                    end
                end
            end
        end

        # print info
        if verbose
            mul!(WH, W, H)
            preobjv = objv
            objv = convert(T, 0.5) * sqL2dist(X, WH)
            @printf("%5d    %12.5e    %12.5e    %12.5e    %8.4f    %12d\n",
                t, objv, objv - preobjv, pgnrm, α, it)
        end
    end
    return H, t
end


## sub-routines for updating W

struct ALSGradUpdW_State{T}
    G::Matrix{T}      # gradient
    Wn::Matrix{T}     # newW in back-tracking
    Wp::Matrix{T}     # previous newW
    D::Matrix{T}      # Wn - W
    HHt::Matrix{T}    # HH' (pre-computed)
    XHt::Matrix{T}    # XH' (pre-computed)
    DHHt::Matrix{T}   # D * HH'

    function ALSGradUpdW_State{T}(X, W, H) where T
        p, k = size(W)
        new{T}(Matrix{T}(undef, p, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, k, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, p, k))
    end
end
ALSGradUpdW_State(X, W::VecOrMat{T}, H::VecOrMat{T}) where {T} = ALSGradUpdW_State{T}(X, W, H)

function set_h!(s::ALSGradUpdW_State, X, H)
    Ht = transpose(H)
    mul!(s.HHt, H, Ht)
    mul!(s.XHt, X, Ht)
end


function alspgrad_updatew!(X,
                           W::VecOrMat{T},
                           H::VecOrMat{T};
                           maxiter::Int = 1000,
                           traceiter::Int = 20,
                           tolg::T = cbrt(eps(T)),
                           beta::T = convert(T, 0.2),
                           sigma::T = convert(T, 0.01),
                           verbose::Bool = false) where T

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
        objv = convert(T, 0.5) * sqL2dist(X, WH)
        @printf("%5d    %12.5e\n", 0, objv)
    end

    # main loop
    t = 0
    converged = false
    decr_alpha = true
    α = one(T) / one(eltype(G))
    while !converged && t < maxiter
        t += 1

        # compute gradient
        mul!(G, W, HHt)
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
                isfinite(α) || error("α is not finite")

                # projected update of W to Wn
                @inbounds for i = 1:length(Wn)
                    wi = W[i]
                    Wn[i] = wi_ = max(wi - α * G[i], zero(T))
                    D[i] = wi_ - wi
                end

                # compute criterion
                dv1 = BLAS.dot(G, D)  # <G, D>
                mul!(DHHt, D, HHt)
                dv2 = BLAS.dot(DHHt, D)  # <D * HHt, D>

                # back-track
                suff_decr = ((1 - σ) * dv1 + convert(T, 0.5) * dv2) < 0

                if it == 1
                    decr_alpha = !suff_decr
                    copyto!(Wp, W)
                end

                if decr_alpha
                    if suff_decr
                        copyto!(W, Wn)
                        break
                    else
                        α *= β
                    end
                else
                    if !suff_decr || isapprox(Wp, Wn, atol=eps(T))
                        copyto!(W, Wp)
                        break
                    else
                        α /= β
                        copyto!(Wp, Wn)
                    end
                end
            end
        end

        # print info
        if verbose
            mul!(WH, W, H)
            preobjv = objv
            objv = convert(T, 0.5) * sqL2dist(X, WH)
            @printf("%5d    %12.5e    %12.5e    %12.5e    %8.4f    %12d\n",
                t, objv, objv - preobjv, pgnrm, α, it)
        end
    end
    return H, t
end


## main algorithm

mutable struct ALSPGrad{T}
    maxiter::Int      # maximum number of main iterations
    maxsubiter::Int   # maximum number of iterations within a sub-routine
    tol::T            # tolerance of changes on W & H (main)
    tolg::T           # tolerance of grad norm in sub-routine
    update_H::Bool    # whether to update H
    verbose::Bool     # whether to show procedural information (main)

    function ALSPGrad{T}(;maxiter::Integer=100,
                          maxsubiter::Integer=200,
                          tol::Real=cbrt(eps(T)),
                          tolg::Real=eps(T)^(1/4),
                          update_H::Bool=true,
                          verbose::Bool=false) where T
        new{T}(maxiter,
               maxsubiter,
               tol,
               tolg,
               update_H,
               verbose)
    end
end

mutable struct ALSPGradUpd{T} <: NMFUpdater{T}
    update_H::Bool
    maxsubiter::Int
    tolg::T
end

solve!(alg::ALSPGrad, X, W, H) =
    nmf_skeleton!(ALSPGradUpd(alg.update_H, alg.maxsubiter, alg.tolg),
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


struct ALSPGradUpd_State{T}
    WH::Matrix{T}
    uhstate::ALSGradUpdH_State{T}
    uwstate::ALSGradUpdW_State{T}

    ALSPGradUpd_State{T}(X, W, H) where {T} =
        new{T}(W * H,
               ALSGradUpdH_State{T}(X, W, H),
               ALSGradUpdW_State{T}(X, W, H))
end

prepare_state(::ALSPGradUpd{T}, X, W, H) where {T} = ALSPGradUpd_State{T}(X, W, H)
evaluate_objv(u::ALSPGradUpd{T}, s::ALSPGradUpd_State{T}, X, W, H) where T = convert(T, 0.5) * sqL2dist(X, s.WH)

function update_wh!(upd::ALSPGradUpd, s::ALSPGradUpd_State, X, W, H)
    T = eltype(W)

    # update H
    if upd.update_H
        set_w!(s.uhstate, X, W)
        iterH = _alspgrad_updateh!(X, W, H, s.uhstate,
            upd.maxsubiter, 20, upd.tolg, convert(T, 0.2), convert(T, 0.01), false)[2]

        if iterH == 1
            upd.tolg *= 0.1
        end
    end

    # update W
    set_h!(s.uwstate, X, H)
    iterW = _alspgrad_updatew!(X, W, H, s.uwstate,
        upd.maxsubiter, 20, upd.tolg, convert(T, 0.2), convert(T, 0.01), false)[2]

    if iterW == 1
        upd.tolg *= 0.1
    end

    # update WH
    mul!(s.WH, W, H)
end