# Coordinate descent method, translated from the Python/Cython implementation
#  in scikit-learn and modified to comply with the interfaces of the NMF package

# Original files
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/nmf.py
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/cdnmf_fast.pyx

# Original implementation authors:
# Vlad Niculae
# Lars Buitinck
# Mathieu Blondel <mathieu@mblondel.org>
# Tom Dupre la Tour

# Original license: BSD 3 clause

# Julia translation: Vilim Štih

# Reference: Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
#  large scale nonnegative matrix and tensor factorizations."
#  IEICE transactions on fundamentals of electronics, communications and
#  computer sciences 92.3: 708-721, 2009.


struct CoordinateDescent{T} <: AbstractNMFAlgorithm
    maxiter::Int           # maximum number of iterations (in main procedure)
    verbose::Bool          # whether to show procedural information
    tol::T                 # tolerance of changes on W and H upon convergence
    update_H::Bool         # whether to update H
    α::T                   # constant that multiplies the regularization terms
    l₁ratio::T             # select whether the regularization affects the components (H), 
                           # the transformation (W), both or none of them 
                           # (:components, :transformation, :both, :none)
    regularization::Symbol # l1 / l2 regularization mixing parameter (in [0; 1])
    shuffle::Bool          # # if true, randomize the order of coordinates in the CD solver

    function CoordinateDescent{T}(;maxiter::Integer=100,
                              verbose::Bool=false,
                              tol::Real=cbrt(eps(T)),
                              update_H::Bool=true,
                              α::Real=zero(T),
                              regularization=:both,
                              l₁ratio::Real=zero(T),
                              shuffle::Bool=false) where T
        maxiter > 1 || throw(ArgumentError("maxiter must be greater than 1."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        new{T}(maxiter, verbose, tol, update_H, α, l₁ratio, regularization, shuffle)
    end
end


solve!(alg::CoordinateDescent{T}, X, W, H; io::IO=stdout, rng::AbstractRNG=default_rng()) where {T} =
    nmf_skeleton!(io, CoordinateDescentUpd{T}(alg.α, alg.l₁ratio, alg.regularization, alg.shuffle, alg.update_H, rng),
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


struct CoordinateDescentUpd{T,R<:AbstractRNG} <: NMFUpdater{T}
    l₁W::T
    l₂W::T
    l₁H::T
    l₂H::T
    shuffle::Bool
    update_H::Bool
    rng::R
    function CoordinateDescentUpd{T}(α::T, l₁ratio::T, regularization::Symbol, shuffle::Bool, update_H::Bool, rng::R) where {T,R<:AbstractRNG}
        αW = zero(T)
        αH = zero(T)

        if (regularization == :both) || (regularization == :components)
            αH = α
        end

        if (regularization == :both) || (regularization == :transformation)
            αW = α
        end

        new{T,R}(αW*l₁ratio,
                 αW*(1-l₁ratio),
                 αH*l₁ratio,
                 αH*(1-l₁ratio),
                 shuffle,
                 update_H,
                 rng)
    end
end

struct CoordinateDescentState{T}
    WH::Matrix{T}
    HHt::Matrix{T}
    XHt::Matrix{T}
    XtW::Matrix{T}

    function CoordinateDescentState{T}(X, W, H) where T
        p, n, k = nmf_checksize(X, W, H)
        new{T}(W * H,
               Matrix{T}(undef, k, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, n, k))
    end
end

prepare_state(::CoordinateDescentUpd{T}, X, W, H) where T = CoordinateDescentState{T}(X, W, H)

function evaluate_objv(::CoordinateDescentUpd{T}, s::CoordinateDescentState{T}, X, W, H) where T
    mul!(s.WH, W, H)
    convert(T, 0.5) * sqL2dist(X, s.WH)
end

"Updates W only"
function _update_coord_descent!(rng::AbstractRNG, s::CoordinateDescentState{T}, X, W, H,
                                l1_reg, l2_reg, shuffle::Bool, W_flag::Bool) where T
    Ht = transpose(H)
    HHt = s.HHt
    mul!(HHt, H, Ht)
    if W_flag
        XHt = s.XHt
    else
        XHt = s.XtW
    end
    mul!(XHt, X, Ht)

    n_components = size(H, 1)
    n_samples = size(W, 1)

    if l2_reg > 0.
        HHt[diagind(HHt)] .+= l2_reg
    end
    if l1_reg > 0.
        XHt .-= l1_reg
    end
    if shuffle
        permutation = randperm(rng, n_components)
    else
        permutation = 1:n_components
    end

    for t in permutation
        for i in 1:n_samples
             # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
            grad = -XHt[i, t]

            for r in 1:n_components
                grad += HHt[t, r] * W[i, r]
            end

            # Hessian
            hess = HHt[t, t]
            if hess != 0
                W[i, t] = max(W[i, t] - grad / hess, zero(grad))
            end
        end
    end
    return
end


function update_wh!(upd::CoordinateDescentUpd{T}, s::CoordinateDescentState{T},
                    X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where T
    # update W
    _update_coord_descent!(upd.rng, s, X, W, H, upd.l₁W, upd.l₂W, upd.shuffle, true)

    # update H
    if upd.update_H
        Wt = transpose(W)
        Ht = transpose(H)
        Xt = transpose(X)
        _update_coord_descent!(upd.rng, s, Xt, Ht, Wt, upd.l₁H, upd.l₂H, upd.shuffle, false)
    end
    return
end
