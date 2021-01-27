# Greedy Coordinate Descent (GreedyCD) algorithm
#
# Author: Takehiro Sano 
#
# Reference: Cho-Jui Hsieh and Inderjit S. Dhillon. "Fast coordinate descent methods 
#  with variable selection for non-negative matrix factorization." In Proceedings of 
#  the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 
#  pp. 1064–1072, 2011.

mutable struct GreedyCD{T}
    maxiter::Int           # maximum number of iterations (in main procedure)
    verbose::Bool          # whether to show procedural information
    tol::T                 # tolerance of changes on W and H upon convergence
    lambda_w::T            # L1 regularization coefficient for W
    lambda_h::T            # L1 regularization coefficient for H

    function GreedyCD{T}(;maxiter::Integer=100,
                          verbose::Bool=false,
                          tol::Real=cbrt(eps(T)),
                          lambda_w::Real=zero(T),
                          lambda_h::Real=zero(T)) where T

        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        lambda_w >= 0 || throw(ArgumentError("lambda_w must be non-negative."))
        lambda_h >= 0 || throw(ArgumentError("lambda_h must be non-negative."))
        new{T}(maxiter, verbose, tol, lambda_w, lambda_h)
    end
end

solve!(alg::GreedyCD{T}, X, W, H) where T = 
    nmf_skeleton!(GreedyCDUpd{T}(alg.lambda_w, alg.lambda_h), X, W, H, alg.maxiter, alg.verbose, alg.tol)


struct GreedyCDUpd{T} <: NMFUpdater{T}
    lambda_w::T
    lambda_h::T
end

struct GreedyCDUpd_State{T}
    WH::Matrix{T}
    PWW::Matrix{T}
    PHH::Matrix{T}
    PXW::Matrix{T}
    PXH::Matrix{T}
    GW::Matrix{T}
    GH::Matrix{T}
    Wnew::Matrix{T}
    Hnew::Matrix{T}
    SW::Matrix{T}
    SH::Matrix{T}
    DW::Matrix{T}
    DH::Matrix{T}
    qW::Vector{Int}
    qH::Vector{Int}
    
    function GreedyCDUpd_State{T}(X, W::Matrix{T}, H::Matrix{T}) where T
        p, n, k = nmf_checksize(X, W, H)
        new{T}(Matrix{T}(undef, p, n),
               Matrix{T}(undef, k, k),
               Matrix{T}(undef, k, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, n, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, n, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, n, k), 
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, n, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, n, k), 
               Vector{Int}(undef, p),
               Vector{Int}(undef, n))
    end
end

prepare_state(::GreedyCDUpd{T}, X, W, H) where T = GreedyCDUpd_State{T}(X, W, H)

function evaluate_objv(u::GreedyCDUpd{T}, s::GreedyCDUpd_State{T}, X, W, H) where T
    mul!(s.WH, W, H)
    r = convert(T, 0.5) * sqL2dist(X, s.WH)
    if u.lambda_w > 0
        r += u.lambda_w * norm(W, 1)
    end
    if u.lambda_h > 0
        r += u.lambda_h * norm(H, 1)
    end
    return r
end

function _update_GreedyCD!(upd::GreedyCDUpd{T}, s::GreedyCDUpd_State{T}, X, 
                           W::AbstractArray{T}, Ht::AbstractArray{T}, W_flag::Bool) where T
    if W_flag
        lambda = upd.lambda_w
        Wnew = s.Wnew
        P = s.PWW
        Z = s.PXW
        G = s.GW
        S = s.SW
        D = s.DW
        q = s.qW
    else
        lambda = upd.lambda_h
        Wnew = s.Hnew
        P = s.PHH
        Z = s.PXH
        G = s.GH
        S = s.SH
        D = s.DH
        q = s.qH
    end
    n_samples, n_components = size(W)

    mul!(P, transpose(Ht), Ht)
    mul!(Z, X, Ht) 
    mul!(G, W, P)
    G .-= Z
    if lambda > zero(T)
        G .+= lambda
    end

    for r in 1:n_components
        for i in 1:n_samples
            S[i, r] = max(zero(T), W[i, r] - G[i, r] / (eps(T) + P[r, r])) - W[i, r]
            D[i, r] = - G[i, r] * S[i, r] - convert(T, 0.5) * P[r, r] * S[i, r]^2
        end
    end

    p_init = convert(T, -1.0)
    for i in 1:n_samples
        # qi = argmax(D[i, :])
        qi = 1
        maxv = D[i, 1]
        for r in 2:n_components
            if D[i, r] > maxv
                qi = r
                maxv = D[i, r]
            end
        end

        q[i] = qi
        p_init = max(p_init, D[i, qi])
    end

    fill!(Wnew, zero(T))
    nu = convert(T, 0.001)

    for i in 1:n_samples
        qi = q[i]
        for _ in 1:n_components^2
            if D[i, qi] < nu * p_init
                break
            end

            Wnew[i, qi] += S[i, qi]

            for r in 1:n_components
                G[i, r] += S[i, qi] * P[qi, r]
            end

            for r in 1:n_components
                S[i, r] = max(zero(T), W[i, r] - G[i, r] / (eps(T) + P[r, r])) - W[i, r]
                D[i, r] = - G[i, r] * S[i, r] - convert(T, 0.5) * P[r, r] * S[i, r]^2
            end

            # qi = argmax(D[i, :])
            qi = 1
            maxv = D[i, 1]
            for r in 2:n_components
                if D[i, r] > maxv
                    qi = r
                    maxv = D[i, r]
                end
            end
        end
        q[i] = qi
    end
    copyto!(W, W + Wnew)
    projectnn!(W) # countermeasure on rounding error
end

function update_wh!(upd::GreedyCDUpd{T}, s::GreedyCDUpd_State{T}, X, W::Matrix{T}, H::Matrix{T}) where T
    # update W
    Ht = transpose(H)
    _update_GreedyCD!(upd, s, X, W, Ht, true)

    # update H
    Xt = transpose(X)
    _update_GreedyCD!(upd, s, Xt, Ht, W, false)
end