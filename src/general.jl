"""
    calculate_separables_nokw([::Type{AT},] fct, sz::NTuple{N, Int}, offset = sz.÷2 .+1, scale = one(real(eltype(AT))),
                                args...; dims = 1:N,
                        all_axes = (similar_arr_type(AT, dims=Val(1)))(undef, sum(sz[[dims...]])), pos=zero(real(eltype(AT))), 
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
+ `dims`:   a vector `[]` of valid dimensions. Only these dimension will be calculated but they are oriented in ND.
+ `all_axes`: if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.
+ `pos`:    a position shifting the indices passed to `fct` in relationship to the `offset`.

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
function calculate_separables_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, 
                                offset = sz.÷2 .+1,
                                scale = one(real(eltype(AT))),
                                args...; dims = 1:N,
                                all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])),
                                pos=zero(real(eltype(AT))),
                                kwargs...) where {AT, N}

    RT = real(float(eltype(AT)))
    offset = isnothing(offset) ? sz.÷2 .+1 : RT.(offset)
    scale = isnothing(scale) ? one(real(eltype(RT))) : RT.(scale)
    start = ntuple((d)->1, N) # 1 .- offset

    # @show typeof(idc)
    dims = [dims...]
    valid_sz = sz[dims]
    # return ((start[dims[d]]:start[dims[d]]+sz[dims[d]]-1) .- pick_n(dims[d], offset))
    # return out_of_place_assing(nothing, d, fct, pick_n(dims[d], scale) .* ((start[dims[d]]:start[dims[d]]+sz[dims[d]]-1) .- pick_n(dims[d], pos)), sz[dims[d]], arg_n(dims[d], args, RT))
    # allocate a contigous memory to be as cash-efficient as possible and dice it up below
    res = ntuple((d) -> reorient((@view all_axes[1+sum(valid_sz[1:d])-sz[dims[d]]:sum(valid_sz[1:d])]), dims[d], Val(N)), lastindex(dims)) # Vector{AT}()

    # @show kwarg_n(dims[1], kwargs)
    # @show arg_n(dims[1], args)
    # @show idc

    # @show extra_args
    # @show offset
    # @show extra_args
    # @show args
    # @show kwargs
    # @show collect(arg_n(dims[1], args))
    # @show (idc, sz[dims[1]], extra_args..., arg_n(dims[1], args)...)
    # res[1][:] .= fct.(idc, sz[dims[1]], arg_n(dims[1], args)...)
    # idc = pick_n(dims[1], scale) .* ((start[dims[1]]:start[dims[1]]+sz[dims[1]]-1) .- pick_n(dims[1], pos))
    # res = in_place_assing!(res, 1, fct, idc, sz[dims[1]], arg_n(dims[1], args))
    #push!(res, collect(reorient(fct.(idc, sz[1], arg_n(1, args)...; kwarg_n(1, kwargs)...), 1, Val(N))))
    # for d = eachindex(dims)

    # toreturn = (in_place_assing!(res, 1, fct, pick_n(dims[1], scale) .* ((start[dims[1]]:start[dims[1]]+sz[dims[1]]-1) .- pick_n(dims[1], pos)), sz[dims[1]], arg_n(dims[1], args, RT)), 
    #         in_place_assing!(res, 2, fct, pick_n(dims[2], scale) .* ((start[dims[2]]:start[dims[2]]+sz[dims[2]]-1) .- pick_n(dims[2], pos)), sz[dims[2]], arg_n(dims[2], args, RT)))

    # idc = start[dims[1]]:start[dims[1]]+sz[dims[1]]-1
    # @show collect(arg_n(dims[1], args, RT))
    # toreturn = (fct.(idc, sz[1], collect(arg_n(dims[1], args, RT))...), ones(1,10))
    toreturn = ntuple((d) -> 
        # idc = pick_n(dims[d], scale) .* ((start[dims[d]]:start[dims[d]]+sz[dims[d]]-1) .- pick_n(dims[d], pos))
        # myaxis = collect(fct.(idc,arg_n(d, args)...)) # no need to reorient
        # extra_args = kwargs_to_args(defaults, kwarg_n(dims[d], kwargs))
        # res[d][:] .= fct.(idc, sz[dims[d]], arg_n(dims[d], args)...)
        in_place_assing!(res, d, fct, pick_n(dims[d], scale) .* ((start[dims[d]]:start[dims[d]]+sz[dims[d]]-1) .- pick_n(dims[d], offset)), sz[dims[d]], arg_n(dims[d], args, RT))
        # AT(out_of_place_assing(res, d, fct, pick_n(dims[d], scale) .* ((start[dims[d]]:start[dims[d]]+sz[dims[d]]-1) .- pick_n(dims[d], offset)), sz[dims[d]], arg_n(dims[d], args, RT)))

        # LazyArray representation of expression
        # push!(res, myaxis)
        , N) # Vector{AT}()
    # @show typeof(toreturn[1])
    # end
    return toreturn
    # return res
end


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
    # @show collect(y)
    _, in_place_assing_pullback = rrule_via_ad(config, out_of_place_assing, res, d, fct, idc, sz1d, args_d)

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
    return y, in_place_assing_pullback # in_place_assing_pullback
end


function calculate_separables(::Type{AT}, fct, sz::NTuple{N, Int}, 
    args...; dims = 1:N,
    all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])),
    defaults=NamedTuple(), pos=zero(real(eltype(AT))),
    offset = sz.÷2 .+1,
    scale = one(real(eltype(AT))),
    kwargs...) where {AT, N}

    extra_args = kwargs_to_args(defaults, kwargs)
    return calculate_separables_nokw(AT, fct, sz, offset, scale, extra_args..., args...; dims=dims, all_axes=all_axes, defaults=defaults, pos=pos, kwargs...)
end

function calculate_separables(fct, sz::NTuple{N, Int},  args...; dims=1:N,
        all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz[[dims...]])),
        defaults=NamedTuple(), pos=zero(real(eltype(DefaultArrType))),
        offset = sz.÷2 .+1,
        scale = one(real(eltype(DefaultArrType))),
        kwargs...) where {N}
    extra_args = kwargs_to_args(defaults, kwargs)
    calculate_separables(DefaultArrType, fct, sz, extra_args..., args...; dims=dims, all_axes=all_axes, pos=pos, offset=offset, scale=scale, kwargs...)
end

# define custom adjoint for calculate_separables
# function ChainRulesCore.rrule(::typeof(calculate_separables), conv, rec, otf)
# end

#

# calculate_separables_nokw(AT, fct, sz, offset, scale, args...; dims=dims, all_axes=all_axes, defaults=defaults, pos=pos, kwargs...)
# function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(calculate_separables_nokw), ::Type{AT}, fct, sz::NTuple{N, Int},
# function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(calculate_separables_nokw), ::Type{AT}, fct, sz::NTuple{N, Int},
#         offset = sz.÷2 .+1, scale=one(real(eltype(AT))), args...; dims = 1:N,
#         all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])),
#         defaults = NamedTuple(),  pos=zero(real(eltype(AT))), kwargs...)  where {AT, N}

#     println("inside calculate_separables_nokw rrule! $(sz), $(dims), $(offset)")
#     # foward pass
#     y = collect(calculate_broadcasted_nokw(AT, fct, sz, offset, scale, args...;
#                                   dims=dims, all_axes=all_axes, defaults = defaults, pos=pos, kwargs...))

#     @show size(y)
#     @show sum(abs2.(y))
#     # extra_args = kwargs_to_args(defaults, kwarg_n(dims[1], kwargs))
#     # res[1][:] .= fct.(idc, sz[dims[1]], extra_args..., arg_n(dims[1], args)...)

# #     d_off_fct = (idx, sz, args...; kwargs...) -> rrule((x) -> 
# #     fct(x, sz, args...; kwargs...), idx)[2](1)[1]
# #     d_scale_fct = (idx, sz, args...; kwargs...) -> idx * rrule((x, sz, args...; kwargs...) ->
# #     fct(x, sz, args...; kwargs...), idx)[2](1)[1]
# #     d_pos_fct = d_off_fct

#     println("calculating fct gradients")
#     # d_fct_dx = (idx, sz, args...; kwargs...) -> rrule_via_ad(config, (x) -> 
#     #             fct(x, sz, args...), idx; kwargs...)[1]
#     d_offset_fct = (idx, sz, args...; kwargs...) -> rrule_via_ad(config, (x) -> 
#                 fct(x, sz, args...), idx; kwargs...)[1]
#     d_scale_fct = (idx, sz, args...; kwargs...) -> idx * rrule_via_ad(config, (x) ->
#                 fct(x, sz, args...), idx;kwargs...)[1]
#     d_pos_fct = d_offset_fct

#     all_grad_axes = copy(all_axes)   # generate a different memory buffer for the gradients

#     function calculate_separables_pullback(dy) # dy is the gradient of the output
#         # multiply by dy?
#         println("calculating separables pullback")
#         @show dy
#         @show length(dy)
#         @show sum(abs.(dy))
#         @show args
#         @show d_offset_fct.(13.3:0.2:15.3, (20,), args...)
#         @show offset
#         @show scale
#         doffset = sum(reshape([dy...], sz) .* collect(calculate_broadcasted_nokw(AT, d_offset_fct, sz, offset, scale, args...; dims=dims, all_axes=all_grad_axes, pos=pos, defaults = defaults, kwargs...)))
#         @show doffset
#         dscale = dy # .* collect(calculate_broadcasted_nokw(AT, d_scale_fct, sz, offset, scale, args...; dims=dims, all_axes=all_axes, pos=pos, defaults = defaults, kwargs...))[:]
#         # dpos = 1 # calculate_separables(AT, d_pos_fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, offset=offset, scale=scale, defaults = defaults, kwargs...)
#         dargs = args;
#         # It should return the gradient of the inputs
#         # println(doffset)

#         # calculate_separables_nokw(AT, fct, sz, offset, scale, args...; dims=dims, all_axes=all_axes, defaults=defaults, pos=pos, kwargs...)
#         return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), doffset, dscale, dargs...,  NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
#     end
#     @show "returning from pullback"
#     return y, calculate_separables_pullback
# end

# foo(a,b;c=42.0)=a*b*c
# function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(calculate_separables), ::Type{AT}, fct, sz::NTuple{N, Int}, args...; dims = 1:N,
# function ZygoteRules._pullback(::ZygoteRules.AContext, A::typeof(Core.kwcall), kwargs, ::typeof(calculate_separables), ::Type{AT}, fct, sz::NTuple{N, Int}, args...)  where {AT, N}
#     # function ZygoteRules._pullback(::ZygoteRules.AContext, A::typeof(Core.kwcall), kwargs, ::typeof(foo), a, b)
#     # retrieve the kwargs 
#     dims =  get(kwargs, :dims, 1:N)  # insert default as we need that in pullback.
#     all_axes = get(kwargs, :all_axes, (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])))  # insert default as we need that in pullback.
#     defaults = get(kwargs, :defaults, NamedTuple())
#     pos = get(kwargs, :pos, zero(real(eltype(AT))))
#     offset = get(kwargs, :offset, sz.÷2 .+1)
#     scale = get(kwargs, :scale, one(real(eltype(AT))))

#     println("kwargs calculate_separables in zygote rule sz:$(sz), dims:$(dims), offset:$(offset) args:$(args)")
#     # foward pass
#     y = calculate_separables(AT, fct, sz, args...; kwargs...)
#     println("calculated y $(typeof(y))")

#     # Use Zygote's _pullback to obtain the gradient function for `fct` in the separable one-d case
#     d_off_fct = (idx, sz, args...; kwargs...) -> Zygote.pullback((x) -> fct(x, sz, args...; kwargs...), idx)[1]
#     d_scale_fct = (idx, sz, args...; kwargs...) -> idx * Zygote.pullback((x) -> fct(x, sz, args...; kwargs...), idx)[1]
#     # config = Type{acontext} # Zygote.ZygoteRuleConfig

#     # d_off_fct = (idx, sz, args...; kwargs...) -> rrule_via_ad(config, (x) -> 
#     #             fct(x, sz, args...), idx; kwargs...)[1]
#     # d_scale_fct = (idx, sz, args...; kwargs...) -> idx * rrule_via_ad(config, (x) ->
#     #             fct(x, sz, args...), idx;kwargs...)[1]
#     d_pos_fct = d_off_fct

#     # c =  get(kwargs, :c, 42.0)  # insert default as we need that in pullback.
#     println("functions defined")
#     function calculate_separables_pullback(dy) # dy is the gradient of the output
#         println("kwargs calculate_separables pulling back")
#         # multiply by dy?
#         # @show d_off_fct(13.3, 20, args...)

#         doffset = haskey(kwargs, :offset) ? calculate_separables(AT, d_off_fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, offset=offset, scale=scale, defaults = defaults, kwargs...) : Zygote.NoTangent()
#         # doffset = collect(doffset)
#         # @show size(.*(doffset...))
#         # @show eltype(.*(doffset...))
#         println("calculated doffset $(typeof(doffset))")
#         dscale = 1 #calculate_separables(AT, d_scale_fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, offset=offset, scale=scale, defaults = defaults,  kwargs...)
#         dpos = 1 # calculate_separables(AT, d_pos_fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, offset=offset, scale=scale, defaults = defaults, kwargs...)
#         dargs = args;
#         dkwargs = (;offset = doffset, scale = dscale, pos = dpos);
#         # It should return the gradient of the inputs
#         # println(doffset)
#         # return (NoTangent(), NoTangent(), NoTangent(), doffset, dargs...,  NoTangent(), NoTangent(), NoTangent(), dpos, doffset, dscale, NoTangent())
#         println("done calculating $(typeof(dkwargs[:offset]))")
#         return nothing, dkwargs, nothing, nothing, nothing, nothing
#     end
#     println("pullback defined")
#     return y, calculate_separables_pullback

#     # function foo_pullback(dy)
#     #     println("kwargs foo pulling back")
#     #     da = dy*b*c
#     #     db = dy*a*c
#     #     dc = haskey(kwargs, :c) ? dy*a*b : NoTangent()
#     #     dkwargs = (;c=dc)
#     #     return nothing, dkwargs, nothing, da, db
#     # end
#     # return y, foo_pullback
# end


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
function calculate_broadcasted(::Type{AT}, fct, sz::NTuple{N, Int}, args...; dims=1:N, 
        all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])),
        pos=zero(real(eltype(DefaultArrType))), 
        operation = *, kwargs...) where {AT, N}
    Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables(AT, fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, kwargs...)...))
end

function calculate_broadcasted(fct, sz::NTuple{N, Int}, args...; dims=1:N,
        all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz[[dims...]])),
        pos=zero(real(eltype(DefaultArrType))),
        operation = *, kwargs...) where {N}
    Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables(DefaultArrType, fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, kwargs...)...))
end

# function calculate_sep_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, args...; dims=1:N, 
#     all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])),
#     pos=zero(real(eltype(DefaultArrType))), operation = *, defaults = nothing, kwargs...) where {AT, N}
#     # defaults should be evaluated here and filled into args...
#     return calculate_separables_nokw(AT, fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, kwargs...)
# end

# function calculate_sep_nokw(fct, sz::NTuple{N, Int}, args...; dims=1:N,
#     all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz[[dims...]])),
#     pos=zero(real(eltype(DefaultArrType))),  operation = *, defaults = nothing, kwargs...) where {N}
#     return calculate_separables_nokw(AT, fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, kwargs...)
# end

### Versions where offst and scale are without keyword arguments
function calculate_broadcasted_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, args...; dims=1:N, 
    all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])),
    pos=zero(real(eltype(DefaultArrType))), operation = *, defaults = nothing, kwargs...) where {AT, N}
    # defaults should be evaluated here and filled into args...
    res = Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables_nokw(AT, fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, kwargs...)...))
    # @show eltype(collect(res))
    return res
end

function calculate_broadcasted_nokw(fct, sz::NTuple{N, Int}, args...; dims=1:N,
    all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz[[dims...]])),
    pos=zero(real(eltype(DefaultArrType))),  operation = *, defaults = nothing, kwargs...) where {N}
    # defaults should be evaluated here and filled into args...
    res = Broadcast.instantiate(Broadcast.broadcasted(operation, calculate_separables_nokw(DefaultArrType, fct, sz, args...; dims=dims, all_axes=all_axes, pos=pos, kwargs...)...))
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
    separable_view{N}(fct, sz, args...; pos=zero(real(eltype(AT))), offset =  sz.÷2 .+1, scale = one(real(eltype(AT))), operation = .*)

creates an `LazyArray` view of an N-dimensional separable function.
Note that this view consumes much less memory than a full allocation of the collected result.
Note also that an N-dimensional calculation expression may be much slower than this view reprentation of a product of N one-dimensional arrays.
See the example below.
    
# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `dims`:         a vector `[]` of valid dimensions. Only these dimension will be calculated but they are oriented in ND.
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
function separable_view(::Type{AT}, fct, sz::NTuple{N, Int}, args...; dims=1:N,
        all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz[[dims...]])),
        operation = *, kwargs...) where {AT, N}
    res = calculate_separables(AT, fct, sz, args...; dims=dims, all_axes = all_axes, kwargs...)
    return LazyArray(@~ operation.(res...)) # to prevent premature evaluation
end

function separable_view(fct, sz::NTuple{N, Int}, args...; dims=1:N,
        all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz[[dims...]])),
        operation = *, kwargs...) where {N}
    separable_view(DefaultArrType, fct, sz::NTuple{N, Int}, args...; dims=dims, all_axes = all_axes, operation=operation, kwargs...)
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
+ `all_axes`:     if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.
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
