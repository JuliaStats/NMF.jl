
# random initialization

function randinit(X::Matrix{Float64}, k::Integer; normalize::Bool=false, zeroh::Bool=false)
    p, n = size(X)

    W = rand(p, k)
    if normalize
        normalize1_cols!(W)
    end

    H = zeroh ? zeros(k, n) : rand(k, n)
    return (W, H)::(Matrix{Float64}, Matrix{Float64})
end

