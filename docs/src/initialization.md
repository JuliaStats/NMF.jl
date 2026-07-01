```@meta
CurrentModule = NMF
```

# Initialization

Every algorithm starts from an initial pair of factors `W` and `H`. A good
initialization speeds up convergence and can steer the optimizer toward a better
local optimum. [`nnmf`](@ref) applies one of these methods automatically through
its `init` keyword, but you can also call them directly to obtain `(W, H)` for
use with [`solve!`](@ref).

Given `X` of size `p×n` and a rank `k`, each initializer returns `W` of size
`p×k` and `H` of size `k×n`. For algorithms that need only `W` initialized
(such as [`ProjectedALS`](@ref)), pass `zeroh=true` to return a zero `H`.

```@docs
randinit
nndsvd
spa
```
