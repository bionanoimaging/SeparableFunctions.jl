"""
    pick_n(n, v)

picks the `n`th value of the vector v or return the scalar v.
"""
function pick_n(n, v::Number)
    v
end

function pick_n(n, v)
    v[n]
    # v[1+((n-1)%lastindex(v))]
end

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
function arg_n(n, args)
    return (pick_n(n, v) for v in args)
end

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

# moved to NDTools:
# """
#     get_real_arr_type(::Type{TA}) where {TA<:AbstractArray}

# returns the same array type but using `(real(eltype()))` as the element type
# """
# function get_real_arr_type(::Type{TA}) where {TA<:AbstractArray}
#     typeof(similar(TA(undef, ntuple(x->0, ndims(TA))), real(eltype(TA))))
# end
 
# """
#     similar_arr_type(::Type{TA}) where {TA<:AbstractArray}

# returns a similar array type but using as TA, but eltype and ndims can be changed.
# """
# function similar_arr_type(::Type{TA}; dims=N, dtype=T) where {T,N, TA<:AbstractArray{T,N}}
#     typeof(similar(TA(undef, ntuple(x->0, N)), dtype, dims))
# end
 
# function similar_arr_type(::Type{TA}; dims=1, dtype=eltype(TA)) where {TA<:AbstractArray}
#     typeof(similar(TA(undef), dtype, dims))
# end
