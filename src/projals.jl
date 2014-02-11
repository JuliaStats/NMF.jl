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

type ProjectedALS
    maxiter::Int
    verbose::Bool
    tol::Float64
    lambda_w::Float64
    lambda_h::Float64

    function ProjectedALS(;maxiter::Integer=100,
                           verbose::Bool=false,
                           tol::Real=1.0e-6,
                           lambda_w::Real=1.0e-6,
                           lambda_h::Real=1.0e-6)

        new(int(maxiter), 
            verbose,
            float64(tol),
            float64(lambda_w), 
            float64(lambda_h))
    end
end

solve!(alg::ProjectedALS, X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64}) =
    nmf_skeleton!(ProjectedALSUpd(alg.lambda_w, alg.lambda_h), 
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


immutable ProjectedALSUpd <: NMFUpdater
    lambda_w::Float64
    lambda_h::Float64
end

immutable ProjectedALSUpd_State
    WH::Matrix{Float64}
    WtW::Matrix{Float64}
    HHt::Matrix{Float64}
    XHt::Matrix{Float64}

    function ProjectedALSUpd_State(X::Matrix{Float64}, W::Matrix{Float64}, H::Matrix{Float64})
        p, n, k = nmf_checksize(X, W, H)
        new(W * H, 
            Array(Float64, k, k), 
            Array(Float64, k, k), 
            Array(Float64, p, k))
    end
end

prepare_state(::ProjectedALSUpd, X, W, H) = ProjectedALSUpd_State(X, W, H)

function evaluate_objv(u::ProjectedALSUpd, s::ProjectedALSUpd_State, X, W, H)
    r = 0.5 * sqL2dist(X, s.WH)
    if u.lambda_w > 0
        r += (0.5 * u.lambda_w) * abs2(normfro(W))
    end
    if u.lambda_h > 0
        r += (0.5 * u.lambda_h) * abs2(normfro(H))
    end
    return r
end

function update_wh!(upd::ProjectedALSUpd, s::ProjectedALSUpd_State, 
                    X::Matrix{Float64}, 
                    W::Matrix{Float64}, 
                    H::Matrix{Float64})

    # fields
    WH::Matrix{Float64} = s.WH
    WtW::Matrix{Float64} = s.WtW
    HHt::Matrix{Float64} = s.HHt
    XHt::Matrix{Float64} = s.XHt
    lambda_w::Float64 = upd.lambda_w
    lambda_h::Float64 = upd.lambda_h

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


