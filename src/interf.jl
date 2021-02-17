# Interface function: nnmf

function nnmf(X::AbstractMatrix{T}, k::Integer;
              init::Symbol=:nndsvdar,
              alg::Symbol=:greedycd,
              maxiter::Integer=100,
              tol::Real=cbrt(eps(T)/100),
              replicates::Integer=1,
              verbose::Bool=false) where T

    eltype(X) <: Number && all(t -> t >= zero(T), X) || throw(ArgumentError("The elements of X must be non-negative."))

    p, n = size(X)
    k <= min(p, n) || throw(ArgumentError("The value of k should not exceed min(size(X))."))

    replicates >= 1 || throw(ArgumentError("The value of replicates must be positive."))

    # determine whether H needs to be initialized
    initH = alg != :projals

    # perform initialization
    if init == :random
        W, H = randinit(X, k; zeroh=!initH, normalize=true)
    elseif init == :nndsvd
        W, H = nndsvd(X, k; zeroh=!initH)
    elseif init == :nndsvda
        W, H = nndsvd(X, k; variant=:a, zeroh=!initH)
    elseif init == :nndsvdar
        W, H = nndsvd(X, k; variant=:ar, zeroh=!initH)
    else
        throw(ArgumentError("Invalid value for init."))
    end

    # choose algorithm
    if alg == :projals
        alginst = ProjectedALS{T}(maxiter=maxiter, tol=tol, verbose=verbose)
    elseif alg == :alspgrad
        alginst = ALSPGrad{T}(maxiter=maxiter, tol=tol, verbose=verbose)
    elseif alg == :multmse
        alginst = MultUpdate{T}(obj=:mse, maxiter=maxiter, tol=tol, verbose=verbose)
    elseif alg == :multdiv
        alginst = MultUpdate{T}(obj=:div, maxiter=maxiter, tol=tol, verbose=verbose)
    elseif alg == :cd
        alginst = CoordinateDescent{T}(maxiter=maxiter, tol=tol, verbose=verbose)
    elseif alg == :greedycd
        alginst = GreedyCD{T}(maxiter=maxiter, tol=tol, verbose=verbose)
    else
        throw(ArgumentError("Invalid algorithm."))
    end

    # run optimization
    ret = solve!(alginst, X, W, H)
    
    for _ in 2:replicates
        if ret.converged
            break
        end
        ret = solve!(alginst, X, ret.W, ret.H)
    end

    return ret
end
