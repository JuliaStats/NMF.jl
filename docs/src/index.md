```@meta
CurrentModule = NMF
```

# NMF.jl

```@docs
NMF
```

Such factorizations are widely used in text mining, image analysis, and
recommendation systems, where the non-negativity constraint yields parts-based,
interpretable factors.

## Installation

```julia
using Pkg
Pkg.add("NMF")
```

## Quick start

The high-level [`nnmf`](@ref) function performs initialization and optimization
in a single call:

```jldoctest
julia> using NMF

julia> X = rand(8, 6);

julia> r = nnmf(X, 3; alg=:multmse, maxiter=50, tol=1.0e-4);

julia> size(r.W), size(r.H)
((8, 3), (3, 6))
```

The returned [`Result`](@ref) holds the factors `r.W` and `r.H` along with
metadata about the run (`r.niters`, `r.converged`, `r.objvalue`).

## How the package is organized

A factorization proceeds in two stages, each backed by a set of tools:

1. **Initialization** produces starting factors `W` and `H`. See
   [Initialization](@ref).
2. **Optimization** refines the factors toward a local optimum. See
   [Algorithms](@ref).

[`nnmf`](@ref) selects and runs both stages for you through its `init` and `alg`
keyword arguments. For finer control you can call an initializer directly and
then run an algorithm in place with [`solve!`](@ref); see [Examples](@ref).

Apart from [`nnmf`](@ref), the package's names are not exported. Refer to them
with the `NMF.` prefix (for example `NMF.solve!`, `NMF.randinit`), which keeps
user code explicit about where each name comes from.

## See also

[NMFMerge](https://github.com/HolyLab/NMFMerge.jl) augments any least-squares
NMF algorithm by merging components.
