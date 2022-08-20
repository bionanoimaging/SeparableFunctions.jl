"""
    pick_n(n, v)

picks the `n`th value of the vector v or the scalar v.
"""
function pick_n(n, v)
    v[1+((n-1)%lastindex(v))]
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

function calculate_separables(::Type{AT}, fct, sz::NTuple{N, Int}, args...; offset=sz.÷2 .+1, scale=one(real(eltype(AT))), kwargs...) where {AT, N}
    start = 1 .- offset
    idc = pick_n(1, scale) .* (start[1]:start[1]+sz[1]-1)
    res = Vector{AT}()
    push!(res, collect(fct.(idc, arg_n(1, args)...; kwarg_n(1, kwargs)...)))
    for d = 2:N
        idc = pick_n(d, scale) .* (start[d]:start[d]+sz[d]-1)
        # myaxis = collect(fct.(idc,arg_n(d, args)...)) # no need to reorient
        myaxis = collect(reorient(fct.(idc, arg_n(d, args)...; kwarg_n(d, kwargs)...), d, Val(N)))
        # LazyArray representation of expression
        push!(res, myaxis)
    end
    return res
end

"""
    separable_view{N}(fct, sz, args...; offset =  sz.÷2 .+1, scale = 1.0, operation = .*)
    creates an array view of an N-dimensional separable function.
    Note that this view consumes much less memory than a full allocation of the collected result.
    Note also that an N-dimensional calculation expression may be much slower than this view reprentation of a product of N one-dimensional arrays.
    See the example below.
    
# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `offset`:       position of the center from which the position is measured
+ `scale`:        defines the pixel size as vector or scalar. Default: 1.0.
+ `operation`:    the separable operation connecting the separable dimensions

# Example:
```julia
julia> fct = (r, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
julia> my_gaussian = separable_view(fct, (6,5), (0.1,0.2),(0.5,1.0))
(6-element Vector{Float64}) .* (1×5 Matrix{Float64}):
 3.99823e-10  2.18861e-9   4.40732e-9   3.26502e-9   8.89822e-10
 1.3138e-5    7.19168e-5   0.000144823  0.000107287  2.92392e-5
 0.00790705   0.0432828    0.0871609    0.0645703    0.0175975
 0.0871609    0.477114     0.960789     0.71177      0.19398
 0.0175975    0.0963276    0.19398      0.143704     0.0391639
 6.50731e-5   0.000356206  0.000717312  0.000531398  0.000144823
```
"""
function separable_view(::Type{TA}, fct, sz::NTuple{N, Int}, args...; operation = *, kwargs...) where {TA, N}
    res = calculate_separables(TA, fct, sz, args...; kwargs...)
    return LazyArray(@~ operation.(res...)) # to prevent premature evaluation
end

function separable_view(fct, sz::NTuple{N, Int}, args...; operation = *, kwargs...) where {N}
    separable_view(DefaultArrType, fct, sz::NTuple{N, Int}, args...; operation=operation, kwargs...)
end

function separable_create(::Type{TA}, fct, sz::NTuple{N, Int}, args...; operation = *, kwargs...) where {TA, N}
    res = calculate_separables(TA, fct, sz, args...; kwargs...)
    operation.(res...)
end

function separable_create(fct, sz::NTuple{N, Int}, args...; operation = *, kwargs...) where {N}
    res = calculate_separables(DefaultArrType, fct, sz, args...; kwargs...)
    operation.(res...)
end
