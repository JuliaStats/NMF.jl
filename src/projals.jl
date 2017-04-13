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

import Base.LinAlg: copytri!
import Base.LAPACK: potrs!, potrf!, potri!

type ProjectedALS{T}
    maxiter::Int
    verbose::Bool
    tol::T
    lambda_w::T
    lambda_h::T

    function ProjectedALS{T}(;maxiter::Integer=100,
                           verbose::Bool=false,
                           tol::Real=cbrt(eps(T)),
                           lambda_w::Real=cbrt(eps(T)),
                           lambda_h::Real=cbrt(eps(T))) where T <: Real

        new(maxiter, verbose, tol, lambda_w, lambda_h)
    end
end

solve!(alg::ProjectedALS, X, W, H) =
    nmf_skeleton!(ProjectedALSUpd(alg.lambda_w, alg.lambda_h),
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


immutable ProjectedALSUpd{T} <: NMFUpdater{T}
    lambda_w::T
    lambda_h::T
end

immutable ProjectedALSUpd_State{T}
    WH::Matrix{T}
    WtW::Matrix{T}
    HHt::Matrix{T}
    XHt::Matrix{T}

    function ProjectedALSUpd_State{T}(X, W::Matrix{T}, H::Matrix{T}) where T <:Real
        p, n, k = nmf_checksize(X, W, H)
        @compat new(W * H,
                    Array{T,2}(k, k),
                    Array{T,2}(k, k),
                    Array{T,2}(p, k))
    end
end

prepare_state{T}(::ProjectedALSUpd{T}, X, W, H) = ProjectedALSUpd_State{T}(X, W, H)

function evaluate_objv{T}(u::ProjectedALSUpd{T}, s::ProjectedALSUpd_State{T}, X, W, H)
    r = convert(T, 0.5) * sqL2dist(X, s.WH)
    if u.lambda_w > 0
        r += (convert(T, 0.5) * u.lambda_w) * abs2(vecnorm(W))
    end
    if u.lambda_h > 0
        r += (convert(T, 0.5) * u.lambda_h) * abs2(vecnorm(H))
    end
    return r
end

function update_wh!{T}(upd::ProjectedALSUpd{T}, s::ProjectedALSUpd_State{T},
                       X,
                       W::Matrix{T},
                       H::Matrix{T})

    # fields
    WH = s.WH
    WtW = s.WtW
    HHt = s.HHt
    XHt = s.XHt
    lambda_w = upd.lambda_w
    lambda_h = upd.lambda_h

    # update H
    adddiag!(At_mul_B!(WtW, W, W), lambda_h)
    At_mul_B!(H, W, X)   # H <- W'X
    pdsolve!(WtW, H)     # H <- inv(WtW) * H
    projectnn!(H)

    # update W
    adddiag!(A_mul_Bt!(HHt, H, H), lambda_w)
    A_mul_Bt!(XHt, X, H)    # XHt <- XH'
    pdrsolve!(XHt, HHt, W)  # W <- XHt * inv(HHt)
    projectnn!(W)

    # update WH
    A_mul_B!(WH, W, H)
end
