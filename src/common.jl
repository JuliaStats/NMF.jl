# common facilities

# tools to check size

function nmf_checksize(X, W::AbstractMatrix, H::AbstractMatrix)

    p = size(X, 1)
    n = size(X, 2)
    k = size(W, 2)

    if !(size(W,1) == p && size(H) == (k, n))
        throw(DimensionMismatch("Dimensions of X, W, and H are inconsistent."))
    end

    return (p, n, k)
end


# algorithm specifications

"""
    AbstractNMFAlgorithm

Supertype for NMF algorithm specifications. Each concrete subtype bundles the
options for one factorization algorithm; construct an instance of a concrete
subtype and run it with [`solve!`](@ref).

Concrete subtypes: [`MultUpdate`](@ref), [`ProjectedALS`](@ref),
[`ALSPGrad`](@ref), [`CoordinateDescent`](@ref), [`GreedyCD`](@ref), and
[`SPA`](@ref).

# Examples

```jldoctest
julia> X = rand(8, 6);

julia> W, H = NMF.randinit(X, 3);

julia> r = NMF.solve!(NMF.GreedyCD{Float64}(maxiter=100), X, W, H);

julia> size(r.W), size(r.H)
((8, 3), (3, 6))
```
"""
abstract type AbstractNMFAlgorithm end

"""
    solve!(alg::AbstractNMFAlgorithm, X, W, H; io=stdout, rng=default_rng()) -> Result

Factorize `X ≈ W*H` using algorithm `alg`, updating `W` and `H` in place and
returning a [`Result`](@ref).

`W` and `H` must be preallocated to sizes `(p, k)` and `(k, n)`, where `X` is
`p×n` and `k` is the rank, and must already be initialized (see
[`randinit`](@ref), [`nndsvd`](@ref), and [`spa`](@ref)). Some algorithms
require both `W` and `H` to be initialized (e.g. [`MultUpdate`](@ref)); others
need only `W` (e.g. [`ProjectedALS`](@ref)). Progress is written to `io` when
`alg` is constructed with `verbose=true`, and `rng` supplies any randomness the
algorithm uses.

# Examples

```jldoctest
julia> X = rand(8, 6);

julia> W, H = NMF.randinit(X, 3);

julia> r = NMF.solve!(NMF.MultUpdate{Float64}(maxiter=50), X, W, H);

julia> size(r.W), size(r.H)
((8, 3), (3, 6))
```
"""
function solve! end


# the result type

"""
    Result{T}

The value returned by [`nnmf`](@ref) and [`solve!`](@ref), holding a
factorization `X ≈ W*H` together with metadata about the run.

# Fields
- `W::Matrix{T}`: the `p×k` left factor.
- `H::Matrix{T}`: the `k×n` right factor.
- `niters::Int`: number of iterations performed.
- `converged::Bool`: whether the algorithm reached its convergence tolerance.
- `objvalue::T`: objective value at the final iteration.

# Examples

```jldoctest
julia> X = rand(8, 6);

julia> r = nnmf(X, 3);

julia> size(r.W), size(r.H)
((8, 3), (3, 6))
```
"""
struct Result{T}
    W::Matrix{T}
    H::Matrix{T}
    niters::Int
    converged::Bool
    objvalue::T

    function Result{T}(W, H, niters, converged, objv) where T
        if size(W, 2) != size(H, 1)
            throw(DimensionMismatch("Inner dimensions of W and H mismatch."))
        end
        new{T}(W, H, niters, converged, objv)
    end
end

Result(W::AbstractMatrix, H::AbstractMatrix, niters, converged, objv) =
    Result{promote_type(eltype(W), eltype(H))}(W, H, niters, converged, objv)


Base.:(==)(A::Result, B::Result) = A.W == B.W && A.H == B.H && A.niters == B.niters && A.converged == B.converged && A.objvalue == B.objvalue
Base.hash(s::Result, h::UInt) = hash(s.objvalue, hash(s.converged, hash(s.niters, hash(s.H, hash(s.W, h + (0x09c9f08cfcba6de3 % UInt))))))

function Base.show(io::IO, r::Result{T}) where T
    p, k = size(r.W)
    n = size(r.H, 2)
    print(io, "Result{", T, "}(", p, '×', n, " ≈ W(", p, '×', k, ")·H(", k, '×', n, "), ",
          "niters=", r.niters, ", converged=", r.converged, ", objvalue=", r.objvalue, ')')
end


# common algorithmic skeleton for iterative updating methods

abstract type NMFUpdater{T} end

function nmf_skeleton!(io::IO, updater::NMFUpdater{T},
                       X, W::Matrix{T}, H::Matrix{T},
                       maxiter::Int, verbose::Bool, tol) where T
    objv = convert(T, NaN)

    # init
    state = prepare_state(updater, X, W, H)
    preW = Matrix{T}(undef, size(W))
    preH = Matrix{T}(undef, size(H))
    if verbose
        start = time()
        objv = evaluate_objv(updater, state, X, W, H)
        @printf(io, "%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(W & H).relchange")
        @printf(io, "%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    t = 0
    while !converged && t < maxiter
        t += 1
        copyto!(preW, W)
        copyto!(preH, H)

        # update H
        update_wh!(updater, state, X, W, H)

        # determine convergence
        converged, dev = stop_condition(W, preW, H, preH, tol)

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = evaluate_objv(updater, state, X, W, H)
            @printf(io, "%5d    %13.6e    %13.6e    %13.6e    %13.6e\n",
                t, elapsed, objv, objv - preobjv, dev)
        end
    end

    if !verbose
        objv = evaluate_objv(updater, state, X, W, H)
    end
    return Result{T}(W, H, t, converged, objv)
end


function stop_condition(W::AbstractArray{T}, preW::AbstractArray, H::AbstractArray, preH::AbstractArray, eps::AbstractFloat) where T
    devmax = zero(T)
    for j in axes(W,2)
        dev_w = sum_w = zero(T)
        for i in axes(W,1)
            dev_w += (W[i,j] - preW[i,j])^2
            sum_w += (W[i,j] + preW[i,j])^2
        end
        dev_h = sum_h = zero(T)
        for i in axes(H,2)
            dev_h += (H[j,i] - preH[j,i])^2
            sum_h += (H[j,i] + preH[j,i])^2
        end
        # A column that is all-zero in both W and preW (a dead component) gives
        # sum==0 and dev==0; its relative change is zero, so guard the 0/0.
        rel_w = sum_w > 0 ? dev_w/sum_w : zero(T)
        rel_h = sum_h > 0 ? dev_h/sum_h : zero(T)
        devmax = max(devmax, sqrt(max(rel_w, rel_h)))
        if sqrt(dev_w) > eps*sqrt(sum_w) || sqrt(dev_h) > eps*sqrt(sum_h)
            return false, devmax
        end
    end
    return true, devmax
end
