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


mutable struct CoordinateDescent{T}
    maxiter::Int           # maximum number of iterations (in main procedure)
    verbose::Bool          # whether to show procedural information
    tol::T                 # tolerance of changes on W and H upon convergence
    α::T                   # constant that multiplies the regularization terms
    l₁ratio::T             # select whether the regularization affects the components (H), 
                           # the transformation (W), both or none of them 
                           # (:components, :transformation, :both, :none)
    regularization::Symbol # l1 / l2 regularization mixing parameter (in [0; 1])
    shuffle::Bool          # # if true, randomize the order of coordinates in the CD solver

    function CoordinateDescent{T}(;maxiter::Integer=100,
                              verbose::Bool=false,
                              tol::Real=cbrt(eps(T)),
                              α::Real=T(0.0),
                              regularization=:both,
                              l₁ratio::Real=zero(T),
                              shuffle::Bool=false) where T
        new{T}(maxiter, verbose, tol, α, l₁ratio, regularization, shuffle)
    end
end


solve!(alg::CoordinateDescent{T}, X, W, H) where {T} =
    nmf_skeleton!(CoordinateDescentUpd{T}(alg.α, alg.l₁ratio, alg.regularization, alg.shuffle),
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


struct CoordinateDescentUpd{T} <: NMFUpdater{T}
    l₁W::T
    l₂W::T
    l₁H::T
    l₂H::T
    shuffle::Bool
    function CoordinateDescentUpd{T}(α::T, l₁ratio::T, regularization::Symbol, shuffle::Bool) where {T}
        αW = zero(T)
        αH = zero(T)

        if (regularization == :both) || (regularization == :components)
            αH = α
        end

        if (regularization == :both) || (regularization == :transformation)
            αW = α
        end

        new{T}(αW*l₁ratio,
               αW*(1-l₁ratio),
               αH*l₁ratio,
               αH*(1-l₁ratio),
               shuffle)
    end
end

mutable struct CoordinateDescentState{T}
    violation::T
    violation_init::Union{Nothing, T}
end

prepare_state(::CoordinateDescentUpd{T}, X, W, H) where {T} =
 CoordinateDescentState(zero(T), nothing)
evaluate_objv(::CoordinateDescentUpd{T}, s::CoordinateDescentState, X, W, H) where {T} = 
    s.violation / (s.violation_init === nothing ? oneunit(T) : s.violation_init)

"Updates W only"
function _update_coord_descent!(X, W, H, l1_reg, l2_reg, shuffle)
    HHt = H * H'
    XHt = X * H'

    n_components = size(H, 1)
    n_samples = size(W, 1)

    if l2_reg > 0.
        HHt[diagind(HHt)] .+= l2_reg
    end
    if l1_reg > 0.
        XHt .-= l1_reg
    end
    if shuffle
        permutation = randperm(n_components)
    else
        permutation = 1:n_components
    end

    violation = zero(eltype(X))

    for t in permutation
        for i in 1:n_samples
             # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
            grad = -XHt[i, t]

            for r in 1:n_components
                grad += HHt[t, r] * W[i, r]
            end

            # projected gradient
            pg = W[i, t] == 0 ? min(zero(grad), grad) : grad
            violation += abs(pg)

            # Hessian
            hess = HHt[t, t]
            if hess != 0
                W[i, t] = max(W[i, t] - grad / hess, zero(grad))
            end
        end
    end
    return violation
end


function update_wh!(upd::CoordinateDescentUpd{T}, s::CoordinateDescentState{T},
    X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where T
    Ht = transpose(H)

    violation = zero(T)
    violation += _update_coord_descent!(X, W, H, upd.l₁W, upd.l₂W, upd.shuffle)
    Wt = transpose(W)
    violation += _update_coord_descent!(PermutedDimsArray(X, (2,1)), Ht, Wt,
    upd.l₁H, upd.l₂H, upd.shuffle)

    s.violation = violation
    if s.violation_init !== nothing
        s.violation_init = violation
    end
end
