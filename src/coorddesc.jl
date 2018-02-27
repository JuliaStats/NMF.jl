# Coordinate descent method after the one in Scikit-learn


mutable struct TransposedArray{T} <: AbstractArray{T,2}
    A :: AbstractArray{T, 2}
end

Base.size(x::TransposedArray) = size(x.A)[end:-1:1]
Base.getindex(x::TransposedArray, I::Vararg{Int, 2}) = fybc

mutable struct CoordinateDescent{T}
    maxiter::Int
    verbose::Bool
    tol::T
    α::T
    l₁ratio::T
    regularization::Symbol
    shuffle::Bool

    function CoordinateDescent{T}(;maxiter::Integer=100,
                              verbose::Bool=false,
                              tol::Real=cbrt(eps(T)),
                              α::Real=T(0.001),
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
    violation_init::Nullable{T}
end

prepare_state(::CoordinateDescentUpd{T}, X, W, H) where {T} =
 CoordinateDescentState(zero(T), Nullable{T}() )
evaluate_objv(::CoordinateDescentUpd{T}, s::CoordinateDescentState, X, W, H) where {T} =
    s.violation / get(s.violation_init, T(1.0))

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
     Ht = PermutedDimsArray(H, (2, 1))

     violation = zero(T)
     violation += _update_coord_descent!(X, W, H, upd.l₁W, upd.l₂W, upd.shuffle)
     Wt = PermutedDimsArray(W, (2, 1))
     violation += _update_coord_descent!(PermutedDimsArray(X, (2,1)), Ht, Wt,
      upd.l₁H, upd.l₂H, upd.shuffle)

    s.violation = violation
    if isnull(s.violation_init)
        s.violation_init = violation
    end
end
