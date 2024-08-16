"""
    get_com(dat::AbstractArray{T, N}, mask, prod_dims=N) where {T, N}

returns the center of mass of a data array `dat` with a mask `mask`. 
"""
function get_com!(dst::AbstractMatrix{T}, dat::AbstractArray{T, N}, mask, prod_dims=N) where {T, N}
    ax = axes(dat)[1:prod_dims]
    ax = ntuple((d) -> reorient(ax[d], Val(d), Val(prod_dims)), Val(prod_dims))
    sum_dat_mask = sum(dat.*mask, dims=1:prod_dims)
    for d in 1:prod_dims
        dst[d,:] .= (sum(ax[d].*dat.*mask, dims=1:prod_dims) ./ sum_dat_mask)[:]
    end
end
function get_com(dat::AbstractArray{T, N}, mask, prod_dims=N) where {T, N}
    dst = similar(dat, (prod_dims, size(dat)[prod_dims+1:end]...))
    get_com!(dst, dat, mask, prod_dims)
    return dst
end

"""
    get_std(dat::AbstractArray{T, N}, t_ctr, mask) where {T, N}

returns the center of mass of a data array `dat` with a mask `mask`. 
"""
function get_std!(dst::AbstractMatrix{T}, dat::AbstractArray{T, N}, mask, t_ctr, prod_dims=N) where {T, N}
    sum_dat_mask = sum(dat.*mask, dims=1:prod_dims)
    for d in 1:prod_dims
        ax = reorient(axes(dat)[d], Val(d), Val(prod_dims))
        ctrs = reshape(t_ctr[d,:], (ones(Integer, prod_dims)..., size(t_ctr)[2:end]...))
        dst[d,:] .= (sqrt.(sum(dat.*mask.*(ax .- ctrs).^2, dims=1:prod_dims) ./ sum_dat_mask))[:]
    end
end
function get_std(dat::AbstractArray{T, N}, mask, t_ctr, prod_dims=N) where {T, N}
    dst = similar(dat, (prod_dims, size(dat)[prod_dims+1:end]...))
    get_std!(dst, dat, mask, t_ctr, prod_dims)
    return dst
end

function get_bg_sum(dat::AbstractArray{T, N}, prod_dims=N) where {T, N}
    sz = size(dat)[1:prod_dims]
    isz = max.(1, sz.-2)
    inner = select_region_view(dat, isz)
    sdat = sum(dat, dims=1:prod_dims)
    sinner = sum(inner, dims=1:prod_dims)
    ninner = prod(isz)
    nedge = prod(sz) - ninner
    bg = (sdat .- sinner)/nedge
    return bg, (sinner .- ninner.*bg) # minimum(dat)
end

function get_intensity(dat::AbstractArray{T, N}, prod_dims=N) where {T, N}
    sz = size(dat)[1:prod_dims]
    # select a 3x3 pixel region and calculate the average intensity in it
    psz = min.(sz, 3)
    peak = select_region_view(dat, psz)
    npeak = prod(psz)
    return sum(peak, dims=1:prod_dims) / npeak # maximum(meas) - offset
end

function gauss_start(meas::AbstractArray{T, N}, rel_thresh=0.2, prod_dims=N; has_covariance=false) where {T, N}
    sz = size(meas)[1:prod_dims]
    bg, sinner = get_bg_sum(meas, prod_dims)
    # just a fletch factor 1.25 for typical spotsize
    i0 = get_intensity(meas, prod_dims) * 1.25
    meas = meas .- bg .- i0 .* rel_thresh
    mymask = meas .> 0

    t_ctr = get_com(meas, mymask, prod_dims)
    σ =  get_std(meas, mymask, t_ctr, prod_dims) .* 1.22 ./ (1-rel_thresh)
    μ =  t_ctr #  .- (sz.÷2 .+1)
    # @show μ
    # @show σ
    # sumpix = sum(meas .* mymask)
    # tosum(apos, dat) = apos .* dat
    # pos = idx(sz)
    # μ = tuple_sum(tosum.(pos, meas.*mymask)) ./ sumpix
    # mysqr(apos, dat) = abs2.(apos) .* dat 
    # pos = idx(size(meas), offset=size(pos).÷2 .+1 .+ μ)
    # σ = 1.0 .* max.(1.0, sqrt.(tuple_sum(mysqr.(pos, meas.*mymask )) ./ sumpix))
    # if has_covariance
    #     σ = [σ..., zeros(((length(μ))*(length(μ)-1))÷2)...]
    #     @show σ
    # end
    maxint = reshape(0.178 .* sinner[:] ./ prod(σ, dims=1)[:], (1, prod(size(sinner))))
    bg = reshape(bg[:], size(maxint))
    start_params = (bg=bg, intensity = maxint, off=μ, args=σ)
    # @show start_params
    return start_params # Fixed(), Positive
end
