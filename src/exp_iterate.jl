# export mul_exp_ikx!  # intentionally not exported, since this is not matching the speed of the separable version

function mul_exp_ikx!(arr::AbstractArray{T,1}, k::NTuple{1,T2}, p0, ::Val{1}) where {T, T2} #  <: Complex
    val = p0
    step = exp(1im*k[1])
    for n in eachindex(arr)
        arr[n] *= val
        val *= step
    end
end

function mul_exp_ikx!(arr::AbstractArray{T, dims}, k::NTuple{dims, T2}, p0, ::Val{dims}) where {dims,T,T2}
    val = p0
    step = exp(1im*k[end])
    allidx = (Colon() for n=1:dims-1)
    for n in axes(arr, dims) # = 1:size(arr, dims)
        subarr = @view arr[allidx..., n]# dropdims(NDTools.slice(arr,dims,n),dims=dims)
        mul_exp_ikx!(subarr, k[1:end-1], val, Val(dims-1))
        val *= step 
    end
end

function mul_exp_ikx!(arr::AbstractArray{T, dims}; shift_by, ctr=size(arr).÷2 .+1) where {dims,T}
    # cis(x*(-typeof(x)(2pi)*shift_by/sz))
    k = -eltype(arr)(2pi).*shift_by./size(arr)
    p0 = exp.(-1im .* sum(k .* (ctr.-1)))
    mul_exp_ikx!(arr, k, p0, Val(dims))
end
