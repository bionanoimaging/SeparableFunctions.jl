""" 
    arg_n(n,args...)
    returns a Tuple of the n^th vector in args

# Example
```julia
```
"""
function arg_n(n,args)
    return (a[n] for a in args)
end

function calculate_separables(::Type{AT}, fct, sz::NTuple{N, Int}, args...; center=sz.÷2 .+1) where {AT, N}
    first_args = arg_n(1, args)
    start = 1 .- center
    idc = start[1]:start[1]+sz[1]-1
    res = Vector{AT}()
    push!(res, collect(fct.(idc, first_args...)))
    for d = 2:N
        idc = start[d]:start[d]+sz[d]-1
        # myaxis = collect(fct.(idc,arg_n(d, args)...)) # no need to reorient
        myaxis = collect(reorient(fct.(idc,arg_n(d, args)...), d, Val(N)))
        # LazyArray representation of expression
        push!(res, myaxis)
    end
    return res
end

"""
    separable_view{N}(fct, sz, args...)
    creates an array view of an N-dimensional separable function.
    Note that this view consumes much less memory than a full allocation of the collected result.
    Note also that an N-dimensional calculation expression may be much slower than this view reprentation of a product of N one-dimensional arrays.
    See the example below.
    
# Arguments:
+ fct: The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
        The first argument of this function is a Tuple corresponding the centered indices.
+ sz: The size of the N-dimensional array to create
+ args...: a list of arguments, each being an N-dimensional vector

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
function separable_view(::Type{TA}, fct, sz::NTuple{N, Int}, args...; center = sz.÷2 .+1) where {TA, N}
    res = calculate_separables(TA, fct, sz, args...; center=center)
    return LazyArray(@~ .*(res...)) # to prevent premature evaluation
end

function separable_view(fct, sz::NTuple{N, Int}, args...; center = sz.÷2 .+1) where {N}
    separable_view(DefaultArrType, fct, sz::NTuple{N, Int}, args...; center=center)
end

function separable_create(::Type{TA}, fct, sz::NTuple{N, Int}, args...; center = sz.÷2 .+1) where {TA, N}
    res = calculate_separables(TA, fct, sz, args...; center=center)
    .*(res...)
end

function separable_create(fct, sz::NTuple{N, Int}, args...; center = sz.÷2 .+1) where {N}
    res = calculate_separables(DefaultArrType, fct, sz, args...; center=center)
    .*(res...)
end
