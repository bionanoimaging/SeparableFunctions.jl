"""
    calculate_separables_nokw([::Type{AT},] fct, sz::NTuple{N, Int}, offset = sz.÷2 .+1, scale = one(real(eltype(AT))),
                                args...; 
                        all_axes = (similar_arr_type(AT, dims=Val(1)))(undef, sum(sz)),  
                           kwargs...) where {AT, N}

creates a list of one-dimensional vectors, which can be combined to yield a separable array. In a way this can be seen as a half-way Lazy operation.
The (potentially heavy) work of calculating the one-dimensional functions is done now but the memory-heavy calculation of the array is done later.
This function is used in `separable_view` and `separable_create`.

#Arguments
+ `AT`:     optional type signfying the array result type. You can for example use `CuArray{Float32}` using `CUDA` to create the views on the GPU.
+ `fct`:    the function to calculate for each axis index (no need for broadcasting!) of this iterable of seperable axes. Note that the first arguments of `fct` have to be the index of this coordinate and the size of this axis. Any further `args` and `nargs` can follow. Often the second argument is not used but it still needs to be present.
+ `sz`:     the size of the result array (when appying the one-D axes)
+ `offset`: specifying the center (zero-position) of the result array in one-based coordinates. The default corresponds to the Fourier-center.
+ `scale`:  multiplies the index before passing it to `fct`
+ `args`:   further arguments which are passed over to the function `fct`.
+ `all_axes`: if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.

#Example:
```julia
julia> fct = (r, sz, sigma)-> exp(-r^2/(2*sigma^2))
julia> gauss_sep = SeparableFunctions.calculate_separables_nokw(Array{Float32}, fct, (6,5), (0.5,1.0), 1.0, 1.0)
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
function calculate_separables_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, 
                                offset = sz.÷2 .+1,
                                scale = one(real(eltype(AT))),
                                args...; 
                                all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
                                kwargs...) where {AT, N}
    RT = real(float(eltype(AT)))
    offset = isnothing(offset) ? sz.÷2 .+1 : RT.(offset)
    scale = isnothing(scale) ? one(real(eltype(RT))) : RT.(scale)
    # start = ntuple((d)->1, N) # 1 .- offset

    # allocate a contigous memory to be as cash-efficient as possible and dice it up below
    res = ntuple((d) -> reorient((@view all_axes[1+sum(sz[1:d])-sz[d]:sum(sz[1:d])]), d, Val(N)), N) # Vector{AT}()

    toreturn = ntuple((d) -> 
        in_place_assing!(res, d, fct, get_1d_ids(d, sz, offset, scale), sz[d], arg_n(d, args, RT))
        , N) 
    return toreturn
    # return res
end

get_1d_ids(d, sz, offset, scale) = pick_n(d, scale) .* ((1:sz[d]) .- pick_n(d, offset))


# a special in-place assignment, which gets its own differentiation rule for the reverse mode 
# to avoid problems with memory-assignment and AD.
function in_place_assing!(res, d, fct, idc, sz1d, args_d)
    res[d][:] .= fct.(idc, sz1d, args_d...)
    return res[d]
end

function out_of_place_assing(res, d, fct, idc, sz1d, args_d)
    # println("oop assign!")
    return reorient(fct.(idc, sz1d, args_d...), Val(d))
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(in_place_assing!), res, d, fct, idc, sz1d, args_d)
    # println("in rrule in_place_assing!")
    y = in_place_assing!(res, d, fct, idc, sz1d, args_d)
    # @show d
    # @show size(y)
    # @show collect(y)
    _, in_place_assing_pullback = rrule_via_ad(config, out_of_place_assing, res, d, fct, idc, sz1d, args_d)

    function debug_dummy(dy)
        println("in debug_dummy") # sz is (10, 20)
        # @show dy # NoTangent()
        # @show size(dy)  # 1st calls: (1, 20) 2nd call: (10, 1)
        myres = in_place_assing_pullback(dy)
        # @show myres[1] # NoTangent()
        # @show myres[2] # NoTangent()
        # @show myres[3] # NoTangent()
        # @show myres[4] # NoTangent()
        # @show myres[5] # 
        # @show size(myres[5]) # 1st calls: (20,) 2nd call: (10,)
        # @show myres[6] # 0.0
        return myres
    end

    # function in_place_assing_pullback(dy) # dy is a tuple of arrays.
    #     println("in in_place_assing_pullback")

    #     # d_idc = mypullback(dy)
    #     @show size(dy[1])
    #     @show size(derivatives) # idc
    #     each_deriv = ntuple((i) -> sum(dy[i] .* derivatives[1]), length(dy))
    #     @show each_deriv
    #     # @show dy[1]
    #     return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), each_deriv, NoTangent(), NoTangent()
    #     # return NoTangent(), each_deriv, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    # end
    return y, in_place_assing_pullback # in_place_assing_pullback # in_place_assing_pullback
end


"""
    calculate_separables([::Type{AT},] fct, sz::NTuple{N, Int}, args...; 
                                    all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
                                    defaults=NamedTuple(), 
                                    offset = sz.÷2 .+1,
                                    scale = one(real(eltype(AT))),
                                    kwargs...) where {AT, N}

creates a list of one-dimensional vectors, which can be combined to yield a separable array. In a way this can be seen as a half-way Lazy operation.
The (potentially heavy) work of calculating the one-dimensional functions is done now but the memory-heavy calculation of the array is done later.
This function is used in `separable_view` and `separable_create`.
For automatic differentiation support (e.g. for opinizing) you should use the `calculate_separables_nokw` function, since it supports AD, whereas the keyword-based version does not.

#Arguments
+ `AT`:     optional type signfying the array result type. You can for example use `CuArray{Float32}` using `CUDA` to create the views on the GPU.
+ `fct`:    the function to calculate for each axis index (no need for broadcasting!) of this iterable of seperable axes. Note that the first arguments of `fct` have to be the index of this coordinate and the size of this axis. Any further `args` and `nargs` can follow. Often the second argument is not used but it still needs to be present.
+ `sz`:     the size of the result array (when appying the one-D axes)
+ `offset`: specifying the center (zero-position) of the result array in one-based coordinates. The default corresponds to the Fourier-center.
+ `scale`:  multiplies the index before passing it to `fct`
+ `args`:   further arguments which are passed over to the function `fct`.
+ `all_axes`: if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.

#Example:
```julia
julia> fct = (r, sz, sigma)-> exp(-r^2/(2*sigma^2))
julia> gauss_sep = SeparableFunctions.calculate_separables_nokw(Array{Float32}, fct, (6,5), (0.5,1.0), 1.0, 1.0)
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
function calculate_separables(::Type{AT}, fct, sz::NTuple{N, Int}, 
    args...; 
    all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
    defaults=NamedTuple(), 
    offset = sz.÷2 .+1,
    scale = one(real(eltype(AT))),
    kwargs...) where {AT, N}

    extra_args = kwargs_to_args(defaults, kwargs)
    return calculate_separables_nokw(AT, fct, sz, offset, scale, extra_args..., args...; all_axes=all_axes, defaults=defaults, kwargs...)
end

function calculate_separables(fct, sz::NTuple{N, Int},  args...; 
        all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz)),
        defaults=NamedTuple(), 
        offset = sz.÷2 .+1,
        scale = one(real(eltype(DefaultArrType))),
        kwargs...) where {N}
    extra_args = kwargs_to_args(defaults, kwargs)
    calculate_separables(DefaultArrType, fct, sz, extra_args..., args...; all_axes=all_axes, offset=offset, scale=scale, kwargs...)
end


"""
    calculate_broadcasted([::Type{TA},] fct, sz::NTuple{N, Int}, args...; offset=sz.÷2 .+1, scale=one(real(eltype(DefaultArrType))), operation = *, kwargs...) where {TA, N}

returns an instantiated broadcasted separable array, which essentially behaves almost like an array yet uses broadcasting. Test revealed maximal speed improvements for this version.
Yet, a problem is that reduce operations with specified dimensions cause an error. However this can be avoided by calling `collect`.

# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `all_axes`: if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.
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
function calculate_broadcasted(::Type{AT}, fct, sz::NTuple{N, Int}, args...; 
        all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
        operation = *, kwargs...) where {AT, N}
    Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables(AT, fct, sz, args...; all_axes=all_axes, kwargs...)...))
end

function calculate_broadcasted(fct, sz::NTuple{N, Int}, args...; 
        all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz)),
        operation = *, kwargs...) where {N}
    Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables(DefaultArrType, fct, sz, args...; all_axes=all_axes, kwargs...)...))
end

# function calculate_sep_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, args...; 
#     all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
#     operation = *, defaults = nothing, kwargs...) where {AT, N}
#     # defaults should be evaluated here and filled into args...
#     return calculate_separables_nokw(AT, fct, sz, args...; all_axes=all_axes, kwargs...)
# end

# function calculate_sep_nokw(fct, sz::NTuple{N, Int}, args...; 
#     all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz)),
#     operation = *, defaults = nothing, kwargs...) where {N}
#     return calculate_separables_nokw(AT, fct, sz, args...; all_axes=all_axes, kwargs...)
# end

### Versions where offst and scale are without keyword arguments
function calculate_broadcasted_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, args...; 
    all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
    operation = *, defaults = nothing, kwargs...) where {AT, N}
    # defaults should be evaluated here and filled into args...
    res = Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables_nokw_hook(AT, fct, sz, args...; all_axes=all_axes, kwargs...)...))
    # @show eltype(collect(res))
    return res
end

function calculate_separables_nokw_hook(::Type{AT}, fct, sz::NTuple{N, Int}, args...; kwargs...) where {AT, N} 
    return calculate_separables_nokw(AT, fct, sz, args...; kwargs...)
end

# function calculate_separables_nokw_hook2()
# end

# to make the code run with Tuples and Vectors alike
function optional_convert(ref_arg::T, val) where {T}
    return T <: AbstractArray ? [val...] : val
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(calculate_separables_nokw_hook), ::Type{AT}, fct, sz::NTuple{N, Int}, args...; kwargs...) where {AT, N}
    # println("in rrule calculate_separables_nokw_hook")
    y = calculate_separables_nokw(AT, fct, sz, args...; kwargs...)
    # @show typeof(y)
    # @show collect(y)
    _, calculate_separables_nokw_pullback = rrule_via_ad(config, calculate_separables_nokw, AT, fct, sz, args...; kwargs...)

    # fct signature is like this: (x, sz, sigma)-> exp(-x^2/(2*sigma^2)   . args has offset, scale, and further args
    ids = get_1d_ids.(1:N, Ref(sz), Ref(args[1]), Ref(args[2]))
    ids_offset_only = get_1d_ids.(1:N, Ref(sz), Ref(args[1]), one(eltype(AT)))
    # wrap each function accordingly, yieldin a tuple of fuctions fcts
    fcts = ntuple((d) -> ((ids, sz, args...) -> fct.(ids, sz, args...)), N)
    # calcuate a pullback for each of the dimensions via broadcast. singlton arguments are automatically broadcasted:
    # Note that ids as also a tuple of dimensions of ranges
    calculate_fct_value_pullbacks = rrule_via_ad.(config, fcts, ids, sz, args[3:end]...)

    # sep_diff = calculate_separables_nokw(AT, calculate_fct_pullbacks)

    function calculate_separables_nokw_hook_pullback(dy)
        # println("in calculate_separables_nokw_hook_pullback") # sz is (10, 20)
        # @show dy 
        # @show typeof(dy)  # Tangent{Any, Tuple{Matrix{Float32}, Matrix{Float32}}}
        # @show size(dy[1])  # (10, 1)
        # @show size(dy[2])  # (1, 20) 
        # myres = calculate_separables_nokw_pullback(dy)
        # @show length(myres) # 7
        # @show myres[1] # NoTangent() # mytype
        # @show myres[2] # NoTangent() # AT
        # @show myres[3] # NoTangent() # fct
        # @show myres[4] # Tangent{Tuple{Int64, Int64}}(0.0f0, 0.0f0) # sz
        # @show myres[5] # Tangent{Tuple{Float32, Float32}}(-2.893550157546997, -0.7438024878501892) # offset
        # both of the derivatives below are based on the ids- (also called x-) derivative. The first two refers to the second return argument being the pullback 
        # x = scale * (ids_raw - offset), so the derivative dx/doffst = - scale, dx/dscale = ids_raw - offset
        # dy(f(x(offset)))/doffset = dy(x)/df * dx/doffset = df(x)/dx * -scale
        dx_dids = ntuple((d)->calculate_fct_value_pullbacks[d][2](one(eltype(AT)))[2], N)
        doffset = optional_convert(args[1], ntuple((d) -> - (pick_n(d, args[2]) * (dy[d][:]' * dx_dids[d]))[1], N)) # ids @ offset the -1 is since the argument of fct is idx-offset
        doffset = length(args[1]) == 1 ? sum(doffset) : doffset
        # @show doffset  
        dscale = optional_convert(args[2], ntuple((d) -> ((@view dy[d][:])' * (ids_offset_only[d] .* dx_dids[d]))[1], N)) # ids @ offset the -1 is since the argument of fct is idx-offset
        dscale = (length(args[2]) == 1) ? sum(dscale) : dscale
        # @show myres[6] # 9.438228607177734 # scale
        # @show dscale
        # dy(f(x, arg)/darg = dy(x)/df * df / darg 
        # dx_dargs = ntuple((d) -> calculate_fct_value_pullbacks[d][2](@view dy[d][:])[3+1], N) 
        # @show dx_dargs
        # dargs = ntuple((argno) -> ntuple((d) -> (ntuple((d) -> calculate_fct_value_pullbacks[d][2](@view dy[d][:])[3+argno], N))[d],
        #             N), # ids @ offset the -1 is since the argument of fct is idx-offset
        #     length(args)-2) 
        # dargs = ntuple((argno) -> ntuple((d) -> calculate_fct_value_pullbacks[d][2](@view dy[d][:])[3+argno], N), length(args)-2) 
        dargs = ntuple((argno) -> optional_convert(args[2+argno], ntuple((d) -> calculate_fct_value_pullbacks[d][2](@view dy[d][:])[3+argno], N)), length(args)-2) 
        # @show myres[7] # Tangent{Tuple{Float64, Float64}}((-4.5020714f0, -4.9361587f0) # sigma
        # @show dargs
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), doffset, dscale, dargs...)
        # return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), myres[5], myres[6], myres[7])
    end

    return y, calculate_separables_nokw_hook_pullback # in_place_assing_pullback # in_place_assing_pullback
end


function calculate_broadcasted_nokw(fct, sz::NTuple{N, Int}, args...; 
    all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz)),
    operation = *, defaults = nothing, kwargs...) where {N}
    # defaults should be evaluated here and filled into args...
    res = Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables_nokw(DefaultArrType, fct, sz, args...; all_axes=all_axes, kwargs...)...))
    # @show eltype(collect(res))
    return res
end

# towards a Gaussian that can also be rotated:
# mulitply with exp(-(x-x0)*(y-y0)/(sigma_xy)) = exp(-(x-x0)/(sigma_xy)) ^ (y-y0) 
# g = gaussian_sep((200, 200), sigma=2.2)
# vg = Base.broadcasted((x,y)->x, collect(g), ones(1,1,10));
# res = similar(g.args[1], size(vg));
# @time gg = accumulate!(*, res, vg, dims=3);  #, init=collect(g)


"""
    separable_view{N}(fct, sz, args...; offset =  sz.÷2 .+1, scale = one(real(eltype(AT))), operation = .*)

creates an `LazyArray` view of an N-dimensional separable function.
Note that this view consumes much less memory than a full allocation of the collected result.
Note also that an N-dimensional calculation expression may be much slower than this view reprentation of a product of N one-dimensional arrays.
See the example below.
    
# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `all_axes`:     if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.
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
function separable_view(::Type{AT}, fct, sz::NTuple{N, Int}, args...; 
        all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
        operation = *, kwargs...) where {AT, N}
    res = calculate_separables(AT, fct, sz, args...; all_axes = all_axes, kwargs...)
    return LazyArray(@~ operation.(res...)) # to prevent premature evaluation
end

function separable_view(fct, sz::NTuple{N, Int}, args...; 
        all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz)),
        operation = *, kwargs...) where {N}
    separable_view(DefaultArrType, fct, sz::NTuple{N, Int}, args...; all_axes = all_axes, operation=operation, kwargs...)
end

"""
    separable_create([::Type{TA},] fct, sz::NTuple{N, Int}, args...; offset =  sz.÷2 .+1, scale = one(real(eltype(AT))), operation = *, kwargs...) where {TA, N}

creates an array view of an N-dimensional separable function including memory allocation and collection.
See the example below.
    
# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `all_axes`:     if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.
+ `offset`:       position of the center from which the position is measured
+ `scale`:        defines the pixel size as vector or scalar. Default: 1.0.
+ `operation`:    the separable operation connecting the separable dimensions

# Example:
```julia
julia> fct = (r, sz, sigma)-> exp(-r^2/(2*sigma^2))
julia> my_gaussian = separable_create(fct, (6,5), (0.5,1.0); )
6×5 Matrix{Float32}:
 3.99823f-10  2.18861f-9   4.40732f-9   3.26502f-9   8.89822f-10
 1.3138f-5    7.19168f-5   0.000144823  0.000107287  2.92392f-5
 0.00790705   0.0432828    0.0871608    0.0645703    0.0175975
 0.0871608    0.477114     0.960789     0.71177      0.19398
 0.0175975    0.0963276    0.19398      0.143704     0.0391639
 6.50731f-5   0.000356206  0.000717312  0.000531398  0.000144823
```
"""
function separable_create(::Type{TA}, fct, sz::NTuple{N, Int}, args...; operation::Function = *, kwargs...)::similar_arr_type(TA, T, Val(N)) where {T, N, TA <: AbstractArray{T}}
    # res = calculate_separables(TA, fct, sz, args...; kwargs...)
    # operation.(res...)
    res = similar(TA, sz)
    res .= calculate_broadcasted(TA, fct, sz, args...; operation=operation, kwargs...)
    return res
end

## the code below seems not type-stable but the code above is. Why?
function separable_create(fct, sz::NTuple{N, Int}, args...; operation::Function = *, kwargs...)::similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(N)) where {N}
    # TT = similar_arr_type(DefaultArrType, Float32, Val(N))
    # res = calculate_separables(similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(N)), fct, sz, args...; kwargs...)
    # return operation.(res...)
    # resT = 
    separable_create(similar_arr_type(DefaultArrType, Float32, Val(N)), fct, sz, args...; operation=operation, kwargs...)
end
