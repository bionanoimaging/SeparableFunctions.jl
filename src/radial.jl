"""
    get_corner_ranges(sz::NTuple{N}; shifted_dims = zeros(Bool, N), inv_dims = zeros(Bool, N), full_dims=zeros(Bool, N)) where {N}

returns a tuple of ranges that can be used for indexing various corners of an N-dimensional dataset.
#Arguments
+ `sz`:     total size of the data
+ `shifted_dims`:   an iterable of Boolean defining which dimension to shift (by half the datasize)
+ `inv_dims`:       an iterable of Boolean defining which which range dimension
+ `full_dims`:      an iterable of Boolean defining which dimension to include fully as a colon (:)
"""
function get_corner_ranges(sz::NTuple{N}; shifted_dims = zeros(Bool, N), inv_dims = zeros(Bool, N), full_dims=zeros(Bool, N)) where {N}
    mymid = sz .÷ 2 .+1
    mylength = sz .- mymid
    (full_dims[d] ? Colon() : 
        (shifted_dims[d] ? (inv_dims[d] ? (sz[d]:-1:mymid[d]+1) : (mymid[d]+1:sz[d])) :
                           (inv_dims[d] ? (mymid[d]-1:-1:mymid[d]-mylength[d]) : (1:mymid[d]))) for d in 1:lastindex(sz))
end

"""
    copy_corners!(arr::AbstractArray{T,N}) where {T,N}

replicates the first N-dimensional quadrant my several mirror operations over the entire array.
The overwrites the entire array with the (mirrored) content of the first quadrant!

#Arguments
+ `arr`:    The array in which the copy operations are performed
+ `speedup_last_dim=true`:  if `true` a one-dimensional assignment trick is used for the last non-singleton dimension.
"""
function copy_corners!(arr::AbstractArray{T,N}; src = nothing, speedup_last_dim=true) where {T,N}
    sz = size(arr)
    if !isnothing(src)
        arr[get_corner_ranges(sz)...] .= src
    end
    src = let 
        if isnothing(src)
            @views arr[get_corner_ranges(sz)...]
        else
            src
        end
    end
    # mirror in the same line
    shifted_dims = (true, zeros(Bool,N-1)...)
    inv_dims = shifted_dims
    @views arr[get_corner_ranges(sz, shifted_dims=shifted_dims)...] .= src[get_corner_ranges(sz, inv_dims=inv_dims)...]
    last_dim = findlast(sz .> 1)
    do_speedup_last_dim = let # speedup makes no sense if the last non-singleton dim has only 2 dimensions
        if sz[last_dim] > 2
            speedup_last_dim
        else
            false
        end
    end
    for d=2:(do_speedup_last_dim ? last_dim-1 : last_dim)
        shifted_dims = Tuple((d == n) ? true : false for n=1:N)
        inv_dims = shifted_dims
        # the dimension which are already compied need to copied in full size.
        full_dims = Tuple((n < d) ? true : false for n=1:N)
        # mirror half to next dimension
        # on CPU this is a tiny bit faster without @view but consumes some memory, but on GPU there is a big advantage with the @views
        @views arr[get_corner_ranges(sz, full_dims=full_dims, shifted_dims=shifted_dims)...] .= arr[get_corner_ranges(sz, full_dims=full_dims, inv_dims=inv_dims)...]
    end
    if do_speedup_last_dim
        copy_last_dim!(arr)
    end
    return arr
end

"""
    copy_last_dim(arr::AbstractArray{T,N}) where{T,N}

mirrors the last dimension exploiting the 1D-nature of the trailing dimension using 1D indexing of the ND array.
However even-size dimensions need special treatment.
"""
function copy_last_dim!(arr::AbstractArray{T,N}) where{T,N}
    sz =size(arr)
    mid_pos = sz.÷2 .+1
    linear = LinearIndices(arr)
    lin_mid = linear[mid_pos...]
    lin_size = linear[sz...] - lin_mid
    # 1D mirror-copy irgnoring the mistakes:  
    @views arr[lin_mid+1:end] .= arr[lin_mid-1:-1:lin_mid-lin_size]
    dims = length(sz)
    if any(iseven.(sz[1:end-1]))
        dim_size = sz .- mid_pos
        for d=1:dims-1
            if iseven(sz[d])
                colons = ((d==nd) ? 1 : Colon() for nd = 1:dims-1)
                # @show collect(colons)
                @views arr[colons..., mid_pos[dims]+1:end] .= arr[colons..., mid_pos[dims]-1:-1:mid_pos[dims]-dim_size[dims]]
                # fix the middle slice (only needed for dimensions > 2)
                if dims>2
                    ranges_from = ((d==nd) ? 1 : mid_pos[nd]-1:-1:mid_pos[nd]-dim_size[nd] for nd = 1:dims-1)
                    ranges_to = ((d==nd) ? 1 : mid_pos[nd]+1:sz[nd] for nd = 1:dims-1)
                    @views arr[ranges_to..., mid_pos[dims]] .= arr[ranges_from..., mid_pos[dims]]
                end
            end
        end
    end
    arr
end

"""
    calc_radial2_symm!(arr::TA, fct; scale = one(real(T)), myrr2sep = rr2_sep(size(arr); scale=scale, offset=size(arr).÷2 .+1)) where {N,T}

evaluates the radial function `fct` over the entire array. The function needs to accept the square of the radius as argument!
The calculation is done fast by only evaluating on the first quadrant and replicating the results by copy operations using `copy_corners!()`.

#Arguments
+ `arr`:    The array into which to evaluate the radial function 
+ `fct`:        The radial function of the squared radius, to be evaluate on the array coordinates. 

+ `scale`:      the vetorized scaling of the pixels (only used, if myrr2sep is not supplied by the user)
+ `myrr2sep`:   The separable xx^2 and yy^2 etc. information as obtained by `rr2_sep()`.
"""
function calc_radial2_symm!(arr::TA, fct; scale = one(real(eltype(TA))), myrr2sep = rr2_sep(real_arr_type(TA), size(arr).÷2 .+1; scale=scale, offset=size(arr).÷2 .+1)) where {TA}
    sz = size(arr)
    # mymid = sz .÷ 2 .+1
    # reduces each of the vectors by two
#    corners = ((sep[((1:size(sep,d)÷2+1) for d=1:lastindex(sep))...] for sep in myrr2sep)
    # @show typeof(arr)
    @views arr[get_corner_ranges(sz)...] .= fct.(myrr2sep)
    # @views arr[1:mymid[1],1:mymid[2]] .= fct.(myrr2sep[1][1:mymid[1],:] .+ myrr2sep[2][:,1:mymid[2]])
    copy_corners!(arr)
end


"""
    calc_radial2_symm([::Type{TA},] sz::NTuple,  fct; scale = one(real(T)), myrr2sep = rr2_sep(size(arr); scale=scale, offset=size(arr).÷2 .+1)) where {N,T}

evaluates the radial function `fct` in a newly created array. The function needs to accept the square of the radius as argument!
The calculation is done fast by only evaluating on the first quadrant and replicating the results by copy operations using `copy_corners!()`.

#Arguments
+ `TA`:         The array type for the newly created array. 
+ `sz`:         The size of the newly created array. 
+ `fct`:        The radial function of the squared radius, to be evaluate on the array coordinates. 

+ `scale`:      the vetorized scaling of the pixels (only used, if myrr2sep is not supplied by the user)
+ `myrr2sep`:   The separable xx^2 and yy^2 etc. information as obtained by `rr2_sep()`.
"""
function calc_radial2_symm(::Type{TA}, sz::NTuple, fct; scale = one(real(eltype(TA))), myrr2sep = rr2_sep(sz.÷2 .+1; scale=scale, offset=sz.÷2 .+1)) where {TA}
    arr = TA(undef, sz)
    calc_radial2_symm!(arr, fct; myrr2sep = myrr2sep)    
end

function calc_radial2_symm(sz::NTuple, fct; scale = one(real(eltype(DefaultArrType))), myrr2sep = rr2_sep(sz.÷2 .+1; scale=scale, offset=sz .÷2 .+1)) 
    calc_radial2_symm(DefaultArrType, sz, fct; myrr2sep = myrr2sep)    
end


"""
    calc_radial_symm!(arr::TA, fct; scale = one(real(T)), myrr2sep = rr2_sep(size(arr); scale=scale, offset=size(arr).÷2 .+1)) where {N,T}

evaluates the radial function `fct` over the entire array. The function needs to accept the radius as argument.
The calculation is done fast by only evaluating on the first quadrant and replicating the results by copy operations using `copy_corners!()`.

#Arguments
+ `arr`:    The array into which to evaluate the radial function 
+ `fct`:        The function of the radius, to be evaluate on the array coordinates. 
+ `scale`:      the vetorized scaling of the pixels (only used, if myrr2sep is not supplied by the user)
+ `myrr2sep`:   The separable xx^2 and yy^2 etc. information as obtained by `rr2_sep()`.
"""
function calc_radial_symm!(arr::TA, fct; scale = one(real(eltype(TA))), myrr2sep = rr2_sep(real_arr_type(TA), size(arr).÷2 .+1; scale=scale, offset=size(arr).÷2 .+1)) where {TA}
    calc_radial2_symm!(arr, (r2)->fct(sqrt(r2)); scale=scale, myrr2sep = myrr2sep)
end

"""
    calc_radial_symm!([::Type{TA},] sz::NTuple,  fct; scale = one(real(T)), myrr2sep = rr2_sep(size(arr); scale=scale, offset=size(arr).÷2 .+1)) where {N,T}

evaluates the radial function `fct` over the entire array. The function needs to accept the radius as argument.
The calculation is done fast by only evaluating on the first quadrant and replicating the results by copy operations using `copy_corners!()`.

#Arguments
+ `TA`:         The array type for the newly created array. 
+ `sz`:         The size of the newly created array. 
+ `fct`:        The function of the radius, to be evaluate on the array coordinates. 
+ `scale`:      the vetorized scaling of the pixels (only used, if myrr2sep is not supplied by the user)
+ `myrr2sep`:   The separable xx^2 and yy^2 etc. information as obtained by `rr2_sep()`.
"""
function calc_radial_symm(::Type{TA}, sz::NTuple, fct; scale = one(real(eltype(TA))), myrr2sep = rr2_sep(sz.÷2 .+1; scale=scale, offset=sz.÷2 .+1)) where {TA}
    calc_radial2_symm(TA, sz, (r2)->fct(sqrt(r2)); scale=scale, myrr2sep = myrr2sep)
end
function calc_radial_symm(sz::NTuple, fct; scale = one(real(eltype(DefaultArrType))), myrr2sep = rr2_sep(sz.÷2 .+1; scale=scale, offset=sz .÷2 .+1)) 
    calc_radial_symm(DefaultArrType, sz, fct; myrr2sep = myrr2sep)    
end


"""
    radial_speedup_ifa([::Type(TA)], rfun, sz, args...; oversample=8f0, method=BSpline(Cubic(Line(OnGrid()))), kwargs...)

calculates a radially symmetric function on an array fast by using interpolation. This version is compatible with the arguments of
generator functions as present in the package `IndexFunArrays.jl`. See `radial_speedup` for a more general version.
Other interpolation `method`s are for example: BSpline(Cubic(Flat(OnGrid()))), BSpline(Cubic(Line(OnGrid()))), BSpline(Quadratic(Line(OnGrid()))), BSpline(Linear())
# Arguments
+ `rfun`:   The radially symmetric function according to the standards as set in `IndexFunArrays.jl`. E.g. `gaussian`. The first argument is the size of the array to generate. 
+ `sz`:     size of the array to calculate
+ `args...`:    further arguments to hand over to `rfun`
+ `oversample`: defines how many times the pre-calculated functions values are oversampled.
+ `method`:     the interpolation method to use. 
+ `kwargs...`:    further keyword arguments to hand over to `rfun`
"""
function radial_speedup_ifa(::Type{TA}, rfun, sz, args...; oversample=8f0, method=BSpline(Cubic(Line(OnGrid()))), kwargs...) where {TA}
    pmax = ceil(Int, sqrt(sum(sz.*sz))*oversample)
    myrad = TA(rfun((pmax,), args...; scale=1/(oversample*oversample), offset=(1,), kwargs...))
    function rad(pos::SVector)
        return SVector(1 + oversample * sqrt(sum(pos .* pos)))
    end
    src = warp(myrad, rad, axes_corner_only(sz); method=method); 
    arr = similar(myrad, sz) 
    # using TA(parent(src)) below allows this to work with CUDA, but the problem is that warp removed the CuArray type.
    copy_corners!(arr,src=parent(src)) # the parent is needed to get rid of the offset-array nature
end

function radial_speedup_ifa(rfun, sz, args...; oversample=8f0, method=BSpline(Cubic(Line(OnGrid()))), kwargs...)
    radial_speedup_ifa(DefaultArrType, rfun, sz, args...; oversample=oversample, method=method, kwargs...) 
end

"""
    radial_speedup([::Type(TA)], rfun, sz, args...; oversample=8f0, method=BSpline(Cubic(Line(OnGrid()))), kwargs...)

calculates a radially symmetric function on an array fast by using interpolation. 
Other interpolation `method`s are for example: BSpline(Cubic(Flat(OnGrid()))), BSpline(Cubic(Line(OnGrid()))), BSpline(Quadratic(Line(OnGrid()))), BSpline(Linear())
See also `calc_radial_symm` which does not use interpolation and is faster for fairly simple functions.
# Arguments
+ `::Type(TA)]`:    optionally the type of the array can be specified.
+ `rfun`:   The radial function `rfun(r)` with radius `r`, to calculate on the array.
+ `sz`:     size of the array to calculate
+ `args...`:    further arguments to hand over to `rfun`
+ `oversample`: defines how many times the pre-calculated functions values are oversampled.
+ `method`:     the interpolation method to use. 
+ `kwargs...`:    further keyword arguments to hand over to `rfun`
"""
function radial_speedup(::Type{TA}, rfun, sz, args...; oversample=8f0, method=BSpline(Cubic(Line(OnGrid()))), kwargs...) where {TA}
    pmax = ceil(Int, sqrt(sum(sz.*sz))*oversample)
    myrad = rfun.(TA(0:pmax-1)/oversample, args...; kwargs...)
    function rad(pos)
        return SVector(1 + oversample * sqrt(sum(pos .* pos)))
    end
    # super slow in CuArrays, due to _get_index calls:
    # src = similar(myrad, sz.÷2 .+1)
    # img = ImageTransformations.box_extrapolation(myrad; method=method)
    # ImageTransformations.warp!(src, img, rad);
    src = warp(myrad, rad, axes_corner_only(sz); method=method);
    arr = similar(myrad, sz)
    # using TA(parent(src)) below allows this to work with CUDA, but the problem is that warp removed the CuArray type.
    copy_corners!(arr,src=parent(src)) # the parent is needed to get rid of the offset-array nature
end

function radial_speedup(rfun, sz, args...; oversample=8f0, method=BSpline(Cubic(Line(OnGrid()))), kwargs...)
    radial_speedup(DefaultArrType, rfun, sz, args...; oversample=oversample, method=method, kwargs...)
end
