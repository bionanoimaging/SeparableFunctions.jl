"""
    calculate_separables([::Type{AT},] fct, sz::NTuple{N, Int}, args...; dims = 1:N, pos=zero(real(eltype(AT))), offset=sz.÷2 .+1, scale=one(real(eltype(AT))), kwargs...) where {AT, N}

creates a list of one-dimensional vectors, which can be combined to yield a separable array. In a way this can be seen as a half-way Lazy operation.
The (potentially heavy) work of calculating the one-dimensional functions is done now but the memory-heavy calculation of the array is done later.
This function is used in `separable_view` and `separable_create`.

#Arguments
+ `AT`:     optional type signfying the array result type. You can for example use `CuArray{Float32}` using `CUDA` to create the views on the GPU.
+ `fct`:    the function to calculate for each axis index (no need for broadcasting!) of this iterable of seperable axes. Note that the first arguments of `fct` have to be the index of this coordinate and the size of this axis. Any further `args` and `nargs` can follow. Often the second argument is not used but it still needs to be present.
+ `sz`:     the size of the result array (when appying the one-D axes)
+ `args`:   further arguments which are passed over to the function `fct`.
+ `dims`:   a vector `[]` of valid dimensions. Only these dimension will be calculated but they are oriented in ND.
+ `pos`:    a position shifting the indices passed to `fct` in relationship to the `offset`.
+ `offset`: specifying the center (zero-position) of the result array in one-based coordinates. The default corresponds to the Fourier-center.
+ `scale`:  multiplies the index before passing it to `fct`

#Example:
```julia
julia> fct = (r, sz, sigma)-> exp(-r^2/(2*sigma^2))
julia> gauss_sep = calculate_separables(fct, (6,5), (0.5,1.0), pos = (0.1,0.2))
2-element Vector{Array{Float32}}:
 [4.4963495f-9; 0.00014774836; … ; 0.1978987; 0.0007318024;;]
 [0.088921614 0.48675224 … 0.726149 0.1978987]
 julia> my_gaussian = .*(gauss_sep...) # this is how to broadcast it
 6×5 Matrix{Float32}:
 3.99823f-10  2.18861f-9   4.40732f-9   3.26502f-9   8.89822f-10
 1.3138f-5    7.19168f-5   0.000144823  0.000107287  2.92392f-5
 0.00790705   0.0432828    0.0871608    0.0645703    0.0175975
 0.0871608    0.477114     0.960789     0.71177      0.19398
 0.0175975    0.0963276    0.19398      0.143704     0.0391639
 6.50731f-5   0.000356206  0.000717312  0.000531398  0.000144823
```
"""
function calculate_separables(::Type{AT}, fct, sz::NTuple{N, Int}, args...; defaults=NamedTuple(), dims = 1:N, pos=zero(real(eltype(AT))), offset=sz.÷2 .+1, scale=one(real(eltype(AT))), kwargs...) where {AT, N}
    start = 1 .- offset
    idc = pick_n(dims[1], scale) .* ((start[dims[1]]:start[dims[1]]+sz[dims[1]]-1) .- pick_n(dims[1], pos))
    # @show typeof(idc)
    dims = [dims...]
    valid_sz = sz[dims]
    all_axes = (similar_arr_type(AT, dims=1))(undef, sum(valid_sz))
    # allocate a contigous memory to be as cash-efficient as possible and dice it up below
    res = ntuple((d) -> reorient((@view all_axes[1+sum(valid_sz[1:d])-sz[dims[d]]:sum(valid_sz[1:d])]), dims[d], Val(N)), lastindex(dims)) # Vector{AT}()
    # @show kwarg_n(dims[1], kwargs)
    # @show arg_n(dims[1], args)
    # @show idc
    extra_args = kwargs_to_args(defaults, kwarg_n(dims[1], kwargs))
    res[1][:] .= fct.(idc, sz[dims[1]], extra_args..., arg_n(dims[1], args)...)
    #push!(res, collect(reorient(fct.(idc, sz[1], arg_n(1, args)...; kwarg_n(1, kwargs)...), 1, Val(N))))
    for d = 2:lastindex(dims)
        idc = pick_n(dims[d], scale) .* ((start[dims[d]]:start[dims[d]]+sz[dims[d]]-1) .- pick_n(dims[d], pos))
        # myaxis = collect(fct.(idc,arg_n(d, args)...)) # no need to reorient
        extra_args = kwargs_to_args(defaults, kwarg_n(dims[d], kwargs))
        res[d][:] .= fct.(idc, sz[dims[d]], extra_args..., arg_n(dims[d], args)...)
        # res[d] .= reorient(fct.(idc, sz[d], arg_n(d, args)...; kwarg_n(d, kwargs)...), d, Val(N))
        # LazyArray representation of expression
        # push!(res, myaxis)
    end
    return res
end

function calculate_separables(fct, sz::NTuple{N, Int}, args...; dims=1:N, pos=zero(real(eltype(DefaultArrType))), offset=sz.÷2 .+1, scale=one(real(eltype(DefaultArrType))), kwargs...) where {N}
    calculate_separables(DefaultArrType, fct, sz, args...; dims=dims, pos=pos, offset=offset, scale=scale, kwargs...)
end


"""
    calculate_broadcasted([::Type{TA},] fct, sz::NTuple{N, Int}, args...; dims=1:N, pos=zero(real(eltype(DefaultArrType))), offset=sz.÷2 .+1, scale=one(real(eltype(DefaultArrType))), operation = *, kwargs...) where {TA, N}

returns an instantiated broadcasted separable array, which essentially behaves almost like an array yet uses broadcasting. Test revealed maximal speed improvements for this version.
Yet, a problem is that reduce operations with specified dimensions cause an error. However this can be avoided by calling `collect`.

# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `dims`:         a vector `[]` of valid dimensions. Only these dimension will be calculated but they are oriented in ND.
+ `offset`:       position of the center from which the position is measured
+ `scale`:        defines the pixel size as vector or scalar. Default: 1.0.
+ `operation`:    the separable operation connecting the separable dimensions

```jldoctest
julia> fct = (r, sz, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
julia> my_gaussian = calculate_broadcasted(fct, (6,5), (0.1,0.2), (0.5,1.0))
Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}}(*, 
(Float32[4.4963495f-9; 0.00014774836; … ; 0.1978987; 0.0007318024;;], 
Float32[0.088921614 0.48675224 … 0.726149 0.1978987]))
julia> collect(my_gaussian)
6×5 Matrix{Float32}:
 3.99823f-10  2.18861f-9   4.40732f-9   3.26502f-9   8.89822f-10
 1.3138f-5    7.19168f-5   0.000144823  0.000107287  2.92392f-5
 0.00790705   0.0432828    0.0871608    0.0645703    0.0175975
 0.0871608    0.477114     0.960789     0.71177      0.19398
 0.0175975    0.0963276    0.19398      0.143704     0.0391639
 6.50731f-5   0.000356206  0.000717312  0.000531398  0.000144823
```
"""
function calculate_broadcasted(::Type{TA}, fct, sz::NTuple{N, Int}, args...; dims=1:N, pos=zero(real(eltype(DefaultArrType))), offset=sz.÷2 .+1, scale=one(real(eltype(DefaultArrType))), operation = *, kwargs...) where {TA, N}
    Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables(TA, fct, sz, args...; dims=dims, pos=pos, offset=offset, scale=scale, kwargs...)...))
end

function calculate_broadcasted(fct, sz::NTuple{N, Int}, args...; dims=1:N, pos=zero(real(eltype(DefaultArrType))), offset=sz.÷2 .+1, scale=one(real(eltype(DefaultArrType))), operation = *, kwargs...) where {N}
    Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables(DefaultArrType, fct, sz, args...; dims=dims, pos=pos, offset=offset, scale=scale, kwargs...)...))
end

"""
    separable_view{N}(fct, sz, args...; pos=zero(real(eltype(AT))), offset =  sz.÷2 .+1, scale = one(real(eltype(AT))), operation = .*)

creates an array view of an N-dimensional separable function.
Note that this view consumes much less memory than a full allocation of the collected result.
Note also that an N-dimensional calculation expression may be much slower than this view reprentation of a product of N one-dimensional arrays.
See the example below.
    
# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `dims`:         a vector `[]` of valid dimensions. Only these dimension will be calculated but they are oriented in ND.
+ `offset`:       position of the center from which the position is measured
+ `scale`:        defines the pixel size as vector or scalar. Default: 1.0.
+ `operation`:    the separable operation connecting the separable dimensions

# Example:
```jldoctest
julia> fct = (r, sz, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
julia> my_gaussian = separable_view(fct, (6,5), (0.1,0.2), (0.5,1.0))
(6×1 Matrix{Float32}) .* (1×5 Matrix{Float32}):
 3.99823e-10  2.18861e-9   4.40732e-9   3.26502e-9   8.89822e-10
 1.3138e-5    7.19168e-5   0.000144823  0.000107287  2.92392e-5
 0.00790705   0.0432828    0.0871609    0.0645703    0.0175975
 0.0871609    0.477114     0.960789     0.71177      0.19398
 0.0175975    0.0963276    0.19398      0.143704     0.0391639
 6.50731e-5   0.000356206  0.000717312  0.000531398  0.000144823
```
"""
function separable_view(::Type{TA}, fct, sz::NTuple{N, Int}, args...; dims=1:N, operation = *, kwargs...) where {TA, N}
    res = calculate_separables(TA, fct, sz, args...; dims=dims, kwargs...)
    return LazyArray(@~ operation.(res...)) # to prevent premature evaluation
end

function separable_view(fct, sz::NTuple{N, Int}, args...; dims=1:N, operation = *, kwargs...) where {N}
    separable_view(DefaultArrType, fct, sz::NTuple{N, Int}, args...; dims=dims, operation=operation, kwargs...)
end

"""
    separable_create([::Type{TA},] fct, sz::NTuple{N, Int}, args...; pos=zero(real(eltype(AT))), offset =  sz.÷2 .+1, scale = one(real(eltype(AT))), operation = *, kwargs...) where {TA, N}

creates an array view of an N-dimensional separable function including memory allocation and collection.
See the example below.
    
# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `dims`:         a vector `[]` of valid dimensions. Only these dimension will be calculated but they are oriented in ND.
+ `offset`:       position of the center from which the position is measured
+ `scale`:        defines the pixel size as vector or scalar. Default: 1.0.
+ `operation`:    the separable operation connecting the separable dimensions

# Example:
```julia
julia> fct = (r, sz, sigma)-> exp(-r^2/(2*sigma^2))
julia> my_gaussian = separable_create(fct, (6,5), (0.5,1.0); pos=(0.1,0.2))
6×5 Matrix{Float32}:
 3.99823f-10  2.18861f-9   4.40732f-9   3.26502f-9   8.89822f-10
 1.3138f-5    7.19168f-5   0.000144823  0.000107287  2.92392f-5
 0.00790705   0.0432828    0.0871608    0.0645703    0.0175975
 0.0871608    0.477114     0.960789     0.71177      0.19398
 0.0175975    0.0963276    0.19398      0.143704     0.0391639
 6.50731f-5   0.000356206  0.000717312  0.000531398  0.000144823
```
"""
function separable_create(::Type{TA}, fct, sz::NTuple{N, Int}, args...; operation = *, kwargs...) where {TA, N}
    res = calculate_separables(TA, fct, sz, args...; kwargs...)
    operation.(res...)
end

function separable_create(fct, sz::NTuple{N, Int}, args...; operation = *, kwargs...) where {N}
    res = calculate_separables(DefaultArrType, fct, sz, args...; kwargs...)
    operation.(res...)
end
