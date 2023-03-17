# Interface function: nnmf

function nnmf(X::AbstractMatrix{T}, k::Integer;
              init::Symbol=:nndsvdar,
              initdata=nothing,
              alg::Symbol=:greedycd,
              maxiter::Integer=100,
              tol::Real=cbrt(eps(T)/100),
              replicates::Integer=1,
              W0::Union{AbstractMatrix{T}, Nothing}=nothing,
              H0::Union{AbstractMatrix{T}, Nothing}=nothing,
              update_H::Bool=true,
              verbose::Bool=false,
              trace::Trace{T}=Trace{T}()) where T

    eltype(X) <: Number && all(t -> t >= zero(T), X) || throw(ArgumentError("The elements of X must be non-negative."))

    p, n = size(X)
    k <= min(p, n) || throw(ArgumentError("The value of k should not exceed min(size(X))."))

    replicates >= 1 || throw(ArgumentError("The value of replicates must be positive."))

    if !update_H && init != :custom
        @warn "Only W will be updated."
    end

    if init == :custom
        W0 !== nothing && H0 !== nothing || throw(ArgumentError("To use :custom initialization, set W0 and H0."))
        eltype(W0) <: Number && all(t -> t >= zero(T), W0) || throw(ArgumentError("The elements of W0 must be non-negative."))
        p0, k0 = size(W0)
        p == p0 && k == k0 || throw(ArgumentError("Invalid size for W0."))
        eltype(H0) <: Number && all(t -> t >= zero(T), H0) || throw(ArgumentError("The elements of H0 must be non-negative."))
        k0, n0 = size(H0)
        k == k0 && n == n0 || throw(ArgumentError("Invalid size for H0."))
    else
        W0 === nothing && H0 === nothing || @warn "Ignore W0 and H0 except for :custom initialization."
    end

    # determine whether H needs to be initialized
    initH = alg != :projals

    # perform initialization
    if init == :random
        W, H = randinit(X, k; zeroh=!initH, normalize=true)
    elseif init == :nndsvd
        W, H = nndsvd(X, k; zeroh=!initH, initdata=initdata)
    elseif init == :nndsvda
        W, H = nndsvd(X, k; variant=:a, zeroh=!initH, initdata=initdata)
    elseif init == :nndsvdar
        W, H = nndsvd(X, k; variant=:ar, zeroh=!initH, initdata=initdata)
    elseif init == :spa
        W, H = spa(X, k)
    elseif init == :custom
        W, H = W0, H0
    else
        throw(ArgumentError("Invalid value for init."))
    end

    # choose algorithm
    if alg == :projals
        alginst = ProjectedALS{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H)
    elseif alg == :alspgrad
        alginst = ALSPGrad{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H)
    elseif alg == :multmse
        alginst = MultUpdate{T}(obj=:mse, maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H)
    elseif alg == :multdiv
        alginst = MultUpdate{T}(obj=:div, maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H)
    elseif alg == :cd
        alginst = CoordinateDescent{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H)
    elseif alg == :greedycd
        alginst = GreedyCD{T}(maxiter=maxiter, tol=tol, verbose=verbose, update_H=update_H)
    elseif alg == :spa
        if init != :spa
            throw(ArgumentError("Invalid value for init, use :spa instead."))
        end
        alginst = SPA{T}(obj=:mse)
    else
        throw(ArgumentError("Invalid algorithm."))
    end

    # run optimization
    if alg == :cd
        ret = solve!(alginst, X, W, H, trace=trace)
    else
        ret = solve!(alginst, X, W, H)
    end

    # replicates
    minobjv = ret.objvalue
    for _ in 2:replicates
        W, H = randinit(X, k; zeroh=!initH, normalize=true)
        tmp = solve!(alginst, X, W, H)
        if minobjv > tmp.objvalue
            ret = tmp
            minobjv = tmp.objvalue
        end
    end

    return ret
end
