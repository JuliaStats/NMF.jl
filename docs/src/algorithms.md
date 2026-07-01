```@meta
CurrentModule = NMF
```

# Algorithms

Each factorization algorithm is represented by a concrete subtype of
[`AbstractNMFAlgorithm`](@ref). You construct an instance to hold the algorithm's
options and then run it with [`solve!`](@ref), which updates `W` and `H` in place
and returns a [`Result`](@ref). [`nnmf`](@ref) does this for you when you pass the
matching `alg` symbol.

Every constructor is keyword-only, and all except [`SPA`](@ref) are parameterized
by the working element type `T` (for example `MultUpdate{Float64}`). Several
defaults depend on `T`, so the type parameter also selects the tolerances.

| `alg` symbol | Type | `W`/`H` initialized |
|:-------------|:-----|:--------------------|
| `:multmse`, `:multdiv` | [`MultUpdate`](@ref) | both |
| `:projals` | [`ProjectedALS`](@ref) | `W` only |
| `:alspgrad` | [`ALSPGrad`](@ref) | both |
| `:cd` | [`CoordinateDescent`](@ref) | both |
| `:greedycd` | [`GreedyCD`](@ref) | both |
| `:spa` | [`SPA`](@ref) | via `init=:spa` |

## Running an algorithm

```@docs
AbstractNMFAlgorithm
solve!
```

## Multiplicative update

```@docs
MultUpdate
```

## Projected alternating least squares

```@docs
ProjectedALS
```

## Alternating least squares with projected gradient descent

```@docs
ALSPGrad
```

## Coordinate descent

```@docs
CoordinateDescent
```

## Greedy coordinate descent

```@docs
GreedyCD
```

## Successive projection algorithm

```@docs
SPA
```
