# Projected Alternating Least Squared method
#
#  Solve the following problem via alternate updating:
#
#     (1/2) * ||X - WH||^2
#   + (lambda_w/2) * ||W||^2
#   + (lambda_h/2) * ||H||^2
#
#  At each iteration, the algorithm updates H and W, holding
#  the other fixed. Particularly, it obtains H and W by solving
#  an unconstrained least square problem and then casting
#  negative entries back to zeros.
#

using LinearAlgebra: copytri!
using LinearAlgebra.LAPACK: potrs!, potrf!, potri!

mutable struct ProjectedALS{T}
    maxiter::Int
    verbose::Bool
    tol::T
    lambda_w::T
    lambda_h::T

    function ProjectedALS{T}(;maxiter::Integer=100,
                              verbose::Bool=false,
                              tol::Real=cbrt(eps(T)),
                              lambda_w::Real=cbrt(eps(T)),
                              lambda_h::Real=cbrt(eps(T))) where T

        new{T}(maxiter, verbose, tol, lambda_w, lambda_h)
    end
end

solve!(alg::ProjectedALS, X, W, H) =
    nmf_skeleton!(ProjectedALSUpd(alg.lambda_w, alg.lambda_h),
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


struct ProjectedALSUpd{T} <: NMFUpdater{T}
    lambda_w::T
    lambda_h::T
end

struct ProjectedALSUpd_State{T}
    WH::Matrix{T}
    WtW::Matrix{T}
    HHt::Matrix{T}
    XHt::Matrix{T}

    function ProjectedALSUpd_State{T}(X, W::Matrix{T}, H::Matrix{T}) where T
        p, n, k = nmf_checksize(X, W, H)
        new{T}(W * H,
               Matrix{T}(undef, k, k),
               Matrix{T}(undef, k, k),
               Matrix{T}(undef, p, k))
    end
end

prepare_state(::ProjectedALSUpd{T}, X, W, H) where {T} = ProjectedALSUpd_State{T}(X, W, H)

function evaluate_objv(u::ProjectedALSUpd{T}, s::ProjectedALSUpd_State{T}, X, W, H) where T
    r = convert(T, 0.5) * sqL2dist(X, s.WH)
    if u.lambda_w > 0
        r += (convert(T, 0.5) * u.lambda_w) * abs2(norm(W))
    end
    if u.lambda_h > 0
        r += (convert(T, 0.5) * u.lambda_h) * abs2(norm(H))
    end
    return r
end

function update_wh!(upd::ProjectedALSUpd{T}, s::ProjectedALSUpd_State{T},
                    X,
                    W::Matrix{T},
                    H::Matrix{T}) where T

    # fields
    WH = s.WH
    WtW = s.WtW
    HHt = s.HHt
    XHt = s.XHt
    lambda_w = upd.lambda_w
    lambda_h = upd.lambda_h

    # update H
    Wt = transpose(W)
    adddiag!(mul!(WtW, Wt, W), lambda_h)
    mul!(H, Wt, X)       # H <- W'X
    pdsolve!(WtW, H)     # H <- inv(WtW) * H
    projectnn!(H)

    # update W
    Ht = transpose(H)
    adddiag!(mul!(HHt, H, Ht), lambda_w)
    mul!(XHt, X, Ht)    # XHt <- XH'
    pdrsolve!(XHt, HHt, W)  # W <- XHt * inv(HHt)
    projectnn!(W)

    # update WH
    mul!(WH, W, H)
end
