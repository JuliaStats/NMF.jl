# Interface function: nnmf

function nnmf(X::AbstractMatrix{T}, k::Integer;
              init::Symbol=:nndsvdar,
              alg::Symbol=:alspgrad,
              maxiter::Integer=100,
              tol::Real=cbrt(eps(T)/100),
              verbose::Bool=false) where T

    p, n = size(X)
    k <= min(p, n) || throw(ArgumentError("The value of k should not exceed min(size(X))."))

    # determine whether H needs to be initialized
    initH = alg == :projals

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
    else
        throw(ArgumentError("Invalid algorithm."))
    end

    # run optimization
    solve!(alginst, X, W, H)
end
