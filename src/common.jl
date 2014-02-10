# common facilities

# the result type

immutable NMFResult
    W::Matrix{Float64}
    H::Matrix{Float64}
    objv::Float64

    function NMFResult(W::Matrix{Float64}, H::Matrix{Float64}, objv::Float64)
        size(W, 2) == size(H, 1) || 
            throw(DimensionMismatch("Inner dimensions of W and H mismatch."))
        new(W, H, objv)
    end
end


