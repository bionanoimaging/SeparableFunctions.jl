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
"""
function copy_corners!(arr::AbstractArray{T,N}) where {T,N}
    sz = size(arr)
    # mirror in the same line
    shifted_dims = (true, zeros(Bool,N-1)...)
    inv_dims = shifted_dims
    @views arr[get_corner_ranges(sz, shifted_dims=shifted_dims)...] .= arr[get_corner_ranges(sz, inv_dims=inv_dims)...]
    for d=2:N
        shifted_dims = Tuple((d == n) ? true : false for n=1:N)
        inv_dims = shifted_dims
        # the dimension which are already compied need to copied in full size.
        full_dims = Tuple((n < d) ? true : false for n=1:N)
        # mirror half to next dimension
        # on CPU this is a tiny bit faster without @view but consumes some memory, but on GPU there is a big advantage with the @views
        @views arr[get_corner_ranges(sz, full_dims=full_dims, shifted_dims=shifted_dims)...] .= arr[get_corner_ranges(sz, full_dims=full_dims, inv_dims=inv_dims)...]
    end
    return arr
end

function get_real_arr_type(::Type{TA}) where {TA}
    typeof(similar(TA(undef, ntuple(x->0, ndims(TA))), real(eltype(TA))))
end
 
"""
    calc_radial_symm!(arr::TA, fct; scale = one(real(T)), offset=size(arr).÷2 .+1, myrr2sep = rr2_sep(size(arr); scale=scale, offset=offset)) where {N,T}

evaluates the radial function `fct` over the entire array. The function needs to accept the square of the radius as argument!
The calculation is done fast by only evaluating on the first quadrant and replicating the results by copy operations using `copy_corners!()`.

#Arguments
+ `arr`:    The array into which to evaluate the radial function 
+ `fct`:        The radial function of the squared radius, to be evaluate on the array coordinates. 

+ `offset`:     the center of the symmetry (only used, if myrr2sep is not supplied by the user)
+ `scale`:      the vetorized scaling of the pixels (only used, if myrr2sep is not supplied by the user)
+ `myrr2sep`:   The separable xx^2 and yy^2 etc. information as obtained by `rr2_sep()`.
"""
function calc_radial_symm!(arr::TA, fct; scale = one(real(T)), offset=size(arr).÷2 .+1, myrr2sep = rr2_sep(get_real_arr_type(TA), size(arr).÷2 .+1; scale=scale, offset=offset)) where {TA}
    sz = size(arr)
    # mymid = sz .÷ 2 .+1
    # reduces each of the vectors by two
#    corners = ((sep[((1:size(sep,d)÷2+1) for d=1:lastindex(sep))...] for sep in myrr2sep)
    # @show typeof(arr)
    @views arr[get_corner_ranges(sz)...] .= fct.(.+(myrr2sep...))
    # @views arr[1:mymid[1],1:mymid[2]] .= fct.(myrr2sep[1][1:mymid[1],:] .+ myrr2sep[2][:,1:mymid[2]])
    copy_corners!(arr)
end


"""
    calc_radial_symm([::Type{TA},] sz::NTuple,  fct; scale = one(real(T)), offset=size(arr).÷2 .+1, myrr2sep = rr2_sep(size(arr); scale=scale, offset=offset)) where {N,T}

evaluates the radial function `fct` in a newly created array. The function needs to accept the square of the radius as argument!
The calculation is done fast by only evaluating on the first quadrant and replicating the results by copy operations using `copy_corners!()`.

#Arguments
+ `TA`:         The array type for the newly created array. 
+ `sz`:         The size of the newly created array. 
+ `fct`:        The radial function of the squared radius, to be evaluate on the array coordinates. 

+ `offset`:     the center of the symmetry (only used, if myrr2sep is not supplied by the user)
+ `scale`:      the vetorized scaling of the pixels (only used, if myrr2sep is not supplied by the user)
+ `myrr2sep`:   The separable xx^2 and yy^2 etc. information as obtained by `rr2_sep()`.
"""
function calc_radial_symm(::Type{TA}, sz::NTuple, fct; scale = one(real(eltype(TA))), offset=sz.÷2 .+1, myrr2sep = rr2_sep(sz.÷2 .+1; scale=scale, offset=offset)) where {TA}
    arr = TA(undef, sz)
    calc_radial_symm!(arr, fct; myrr2sep = myrr2sep)    
end

function calc_radial_symm(sz::NTuple, fct; scale = one(real(eltype(DefaultArrType))), offset=sz.÷2 .+1, myrr2sep = rr2_sep(sz.÷2 .+1; scale=scale, offset=offset)) where {TA}
    calc_radial_symm(DefaultArrType, sz, fct; myrr2sep = myrr2sep)    
end
