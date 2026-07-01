```@meta
CurrentModule = NMF
```

# High-level interface

[`nnmf`](@ref) runs a full factorization — initialization followed by
optimization — and returns a [`Result`](@ref). The `init` and `alg` keyword
arguments choose the initialization method and factorization algorithm; the
[Initialization](@ref) and [Algorithms](@ref) pages describe the available
choices in detail.

```@docs
nnmf
Result
```
