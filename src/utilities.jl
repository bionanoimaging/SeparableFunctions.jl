"""
    get_vec_dim(vec::AbstractVector{T}, dim, tsz) where {T}

retrieves the coefficients of an argument corresponding to dimension dim.
If this is a vector of numbers or a vector of vectors, the simple index access is returned.
However, if this is a higher-dimensional array, the dimension dim is selected and reshaped to the size of the result.

Parameters:
- `vec`: the vector or array to be accessed
- `dim`: the dimension to be accessed
- `tsz`: the size tuple of the reference array (just the array size without the replications!).
         This is needed to reshape / reorient the result to the correct size.
"""
# function get_vec_dim(vec::AbstractVector{T}, dim, tsz) where {T}
#     # the collect is for CUDA allowscalar
#     dim = length(vec) > 1 ? dim : 1
#     # to keep CuArrays on the device
#     return @view vec[dim:dim]
# end

function get_vec_dim(vec::Tuple, dim, tsz) 
    return vec[dim]
end

"""
    get_vec_dim(arr::AbstractArray{T, N}, dim, tsz) where {T, N}

this version assumes that the argument vector `arr` describes the dimensions
beyond the dimensions of the reference array as described by the size tuple `tsz`.
"""
function get_vec_dim(arr::AbstractArray{T, N}, dim, tsz) where {T, N}
    dim = size(arr,1) > 1 ? dim : 1
    nsz = ntuple((d)-> (d<=length(tsz)) ? 1 : size(arr, d-length(tsz)+1), Val(length(tsz)+N-1))
    # returned is just a (reoriented) view:
    return reshape(selectdim(arr, 1, dim), nsz)
end

function get_vec_dim(num::Number, dim, tsz) 
    return num
end

# since this is also called with empty arguments, we need to handle this case:
function get_vec_dim(num::Nothing, dim, tsz) 
    return []
end

# to make the code run with Tuples and Vectors alike
# This conversion undoes the get_vec_dim operation
# function optional_convert(ref_arg::AbstractVector{T}, val) where {T}
#     return sum.(val)
# end
function optional_convert(ref_arg::AbstractArray{T,N}, val) where {T,N}
    if (prod(size(ref_arg)) == 1)
        return sum(sum.(val))
    end
    res = similar(ref_arg)
    d=1
    for v in val
        dim = size(ref_arg, 1) > 1 ? d : 1
        dv = selectdim(res, 1, dim)
        if (prod(size(dv)) == 1)
            dv .= sum(v[:])
        else
            dv[:] .= v[:]
        end
        d += 1
    end
    return res
end

function optional_convert(ref_arg::NTuple, val::NTuple)
    return sum.(val)
end

function optional_convert(ref_arg::Number, val::NTuple)
    return sum(sum.(val))
end

# Versions that assign in place
function optional_convert_assign!(dst, ref_arg::AbstractArray{T,N}, val) where {T,N}
    if (prod(size(ref_arg)) == 1)
        dst .= sum(sum.(val))
        return
    end
    d=1
    for v in val
        dv = selectdim(dst, 1, d)
        if (prod(size(dv)) == 1)
            dv .= sum(v[:])
        else
            dv[:] .= v[:]
        end
        d += 1
    end
end

function optional_convert_assign!(dst, ref_arg::NTuple, val::NTuple)
    dst .= sum.(val)
end

function optional_convert_assign!(dst, ref_arg::Number, val::NTuple)
    dst = sum(val)
end


"""
    get_arg_sz(sz, args...)

estimates the size of the arguments.
This is useful for the memory allocation function, which requires the extra size of the arguments.

"""
function get_arg_sz(sz, args...)
    max(size.(get_vec_dim.(args, 1, Ref(sz)))...)[length(sz)+1:end]
end


"""
    get_sep_mem(::Type{AT}, sz::NTuple{N, Int}, hyper_sz=(1,)) where {AT, N}

allocates a contingous memory for the separable functions. This is useful if you want to use the same memory for multiple calculations.
It should be passed to the `calculate_separables_nokw` function via the all_axes argument.

Parameters:
- AT: Array type. Can also be a CuArray
- `sz`: size of the separable part of the data. THis does not include the hyperplanes.
- `hyper_sz`: specifies the size of hyperplanes to pre-allocate as outr dimensions to `sz`. If vectors such as `offset` or `scale` specify vectors oriented along any other directions.
 These will automatically be applied differently to each such hyperplane. E.g.: offfset=[reshape([1.0,2.0,3.0],(1,1,3)),reshape([-1.0,1.0,-2.0],(1,1,3))]
 for a 2D sz=(512,512) and 3 hyperplanes.
"""
function get_sep_mem(::Type{AT}, sz::NTuple{N, Int}, hyper_sz=(1,)) where {AT, N}
    hyperplanes = prod(hyper_sz)
    all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, hyperplanes*sum(sz))
    D = length(sz)+ length(hyper_sz)
    dsizes = ntuple((d) -> ntuple((d2)-> (d2 <= N) ? ifelse(d2==d, sz[d], 1) : hyper_sz[d2-N], Val(D)), Val(N))
    res = ntuple((d) -> reshape((@view all_axes[1+hyperplanes*sum(sz[1:d])-hyperplanes*sz[d]:hyperplanes*sum(sz[1:d])]), dsizes[d]), Val(N)) # Vector{AT}()
    # res = ntuple((d) -> reorient((@view all_axes[1+hyperplanes*sum(sz[1:d])-hyperplanes*sz[d]:hyperplanes*sum(sz[1:d])]), Val(d), Val(N)), Val(N)) # Vector{AT}()
    return res
end

"""
    get_bc_mem(::Type{AT}, sz::NTuple{N, Int}, operator, hyper_sz=(1,)) where {AT, N}

allocates a contigous memory block for the separable functions and wraps it into an instantiate broadcast (bc) structure including the bc-`operator`.
This structure is also returned by functions like `gaussian_sep` and can  be reused by supplying it via the
keyword argument `all_axes`. To obtain the bc-operator for predefined functions use `get_operator(fct)` with `fct`
being the `raw_` version of the function, e.g. `get_operator(gassian_raw)`
"""
function get_bc_mem(::Type{AT}, sz::NTuple{N, Int}, operator, hyper_sz=(1,)) where {AT, N}
    return Broadcast.instantiate(Broadcast.broadcasted(operator, get_sep_mem(AT, sz, hyper_sz)...))
end



"""
    pick_n(n, v)

picks the `n`th value of the vector v or return the scalar v.
"""
pick_n(n, v::Number) = v
pick_n(n, v::Tuple) = v[n]
pick_n(n, v::Vector) = v[n]
# for subarrays and alike:
pick_n(n, v) = v[n]

""" 
    arg_n(n, args...)

returns a modified version of args, such that scalars remain scalar but in vectors the n'th position is picked.
This is useful for calling separable functions with their scalar arguments differing for each vector entry. 

# Example
```jdoctest
julia> args = (1,(4,5))
(1, (4, 5))
julia> collect(SeparableFunctions.arg_n(2, args))
2-element Vector{Int64}:
 1
 5
 ```
"""
function arg_n(n, args, T::Type, sz)
    return ntuple((p)->T.(get_vec_dim(args[p], n, sz)), length(args))
    # return ntuple((p)->T(pick_n(n, args[p])), length(args))
end
function arg_n(n, args, sz)
    return ntuple((p) -> get_vec_dim(args[p], n, sz), length(args))
    # return ntuple((p) -> pick_n(n, args[p]), length(args))
end

# function crunch_args(args)
#     return Tuple(pick_n(n, v) for v in args)
# end

""" 
    kwarg_n(n, args...)

returns a modified version of keyword-args kwargs, such that scalar values remain scalar but in vectors the n'th position is picked.
This is useful for calling separable functions with their scalar arguments differing for each vector entry. 

    # Example
```jdoctest
julia> kw = (a=1,b=(4,5))
(a = 1, b = (4, 5))
julia> SeparableFunctions.kwarg_n(2, kw)
(a = 1, b = 5)
```
"""
function kwarg_n(n, kwargs)
    # only change the values of the named tuple but keep the keys (names)
    (;zip(keys(kwargs), arg_n(n, values(kwargs)))...)
end

"""
    kwargs_to_args(defaults, kwargs)

converts key word args to normal args by filling in the default values, if "nothing" is provided.
"""
function kwargs_to_args(defaults, kwargs)
    # ensure that all arguments are valid
    for k in keys(kwargs)
        if !(k in keys(defaults))
            @show defaults
            error("unknown key word argument: $k, possible arguments are: $(defaults).")
        end
    end
    res = []
    for (k,v) in zip(keys(defaults), values(defaults))
        if k in keys(kwargs)
            if !isnothing(k) # do not submit this argument, if the default is `nothing`.
                v = kwargs[k]
                push!(res, v)
            end
        else
            if !isnothing(v) # do not submit this argument, if the default is `nothing`.
                push!(res, v)
            end
        end
    end
    Tuple(res)
end


"""
    broadcast_reduce(f, op, A...; dims, init)

broadcasts the function `f` over the arrays `A` and reduces the result with the operation `op` but requiring less memory.

modified from code by @torrence:
https://discourse.julialang.org/t/mapreduce-with-broadcasting/77053/7?u=rainerheintzmann
"""
function broadcast_reduce(f, op, A...; dims, init)
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, A...))
    return reduce(op, bc; dims, init)
end