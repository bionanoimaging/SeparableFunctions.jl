"""
    calculate_separables_nokw([::Type{AT},] fct, sz::NTuple{N, Int}, offset = sz.÷2 .+1, scale = one(real(eltype(AT))),
                                args...;all_axes = get_sep_mem(AT, sz), kwargs...) where {AT, N}

creates a list of one-dimensional vectors, which can be combined to yield a separable array. In a way this can be seen as a half-way Lazy operator.
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
julia> fct = (r, sz, sigma)-> exp.(.-r.^2/(2*sigma^2))
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
                                offset = nothing,
                                scale = nothing,
                                args...; 
                                all_axes = get_sep_mem(AT, sz, get_arg_sz(sz, offset, scale, args...)),
                                kwargs...) where {AT, N}
    RT = real(float(eltype(AT)))
    RAT = similar_arr_type(AT, RT, Val(1))
    offset = isnothing(offset) ? (sz.÷2 .+1 ) : RT.(offset)
    scale = isnothing(scale) ? RAT([one(RT)]) : RT.(scale)

    # offset = ntuple((d) -> pick_n(d, offset), Val(N))
    # scale = ntuple((d) -> pick_n(d, scale), Val(N))

    # below the cast of the indices is needed to make CuArrays work
    # for (res, d) in zip(all_axes, 1:N)
    #     in_place_assing!(res, d, fct, get_1d_ids(d, sz, offset, scale), sz[d], arg_n(d, args, RT))
    # end
    # idcs = ntuple((d) -> scale[d] .* ((1:sz[d]) .- offset[d]), Val(N))
    # args_1d = ntuple((d) -> arg_n(d, args), Val(N))
    # in_place_assing!.(all_axes, 1, fct, idcs, sz, args_1d)
    for (res, sz1d, d) in zip(all_axes, sz, 1:N)
        off = get_vec_dim(offset, d, sz)
        sca = get_vec_dim(scale, d, sz)
        idc = get_1d_ids(d, sz, offset, scale)
        args_d = arg_n(d, args, RT, sz) # 
        # in_place_assing!(res, 1, fct, idc, sz1d, args_d)
        res .= fct.(idc, sz1d, args_d...) # 5 allocs, 160 bytes
    end
    return all_axes
    # return res
end

get_1d_ids(d, sz::NTuple{N, Int}, offset, scale) where {N} = get_vec_dim(scale, d, sz) .* (reorient((1:sz[d]), d, Val(N)) .- get_vec_dim(offset, d, sz))
get_1d_ids(d, sz::NTuple{N, Int}, offset) where {N} = (reorient((1:sz[d]), d, Val(N)) .- get_vec_dim(offset, d, sz))
# get_1d_ids(d, sz, offset, scale) = pick_n(d, scale) .* ((1:sz[d]) .- pick_n(d, offset))
# get_1d_ids(d, sz, offset::NTuple, scale::NTuple) = scale[d] .* ((1:sz[d]) .- offset[d])

# a special in-place assignment, which gets its own differentiation rule for the reverse mode 
# to avoid problems with memory-assignment and AD.
function in_place_assing!(res, d, fct, idc, sz1d, args_d)
    res[:] .= fct.(idc, sz1d, args_d...) # 5 allocs, 256 bytes
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
                                    all_axes = get_sep_mem(AT, sz),
                                    defaults=NamedTuple(), 
                                    offset = sz.÷2 .+1,
                                    scale = one(real(eltype(AT))),
                                    kwargs...) where {AT, N}

creates a list of one-dimensional vectors, which can be combined to yield a separable array. In a way this can be seen as a half-way Lazy operator.
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
julia> fct = (r, sz, sigma)-> exp.(.-r.^2/(2*sigma^2))
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
    defaults=NamedTuple(), 
    offset = sz.÷2 .+1,
    scale = one(real(eltype(AT))),
    kwargs...) where {AT, N}

    extra_args = kwargs_to_args(defaults, kwargs)
    return calculate_separables_nokw(AT, fct, sz, offset, scale, extra_args..., args...; defaults=defaults, kwargs...)
end

function calculate_separables(fct, sz::NTuple{N, Int},  args...; 
        defaults=NamedTuple(), 
        offset = sz.÷2 .+1,
        scale = one(real(eltype(DefaultArrType))),
        kwargs...) where {N}
    extra_args = kwargs_to_args(defaults, kwargs)
    calculate_separables(DefaultArrType, fct, sz, extra_args..., args...; offset=offset, scale=scale, kwargs...)
end


"""
    calculate_broadcasted([::Type{TA},] fct, sz::NTuple{N, Int}, args...; offset=sz.÷2 .+1, scale=one(real(eltype(DefaultArrType))), operator = *, kwargs...) where {TA, N}

returns an instantiated broadcasted separable array, which essentially behaves almost like an array yet uses broadcasting. Test revealed maximal speed improvements for this version.
Yet, a problem is that reduce operators with specified dimensions cause an error. However this can be avoided by calling `collect`.

# Arguments:
+ `fct`:          The separable function, with a number of arguments corresponding to `length(args)`, each corresponding to a vector of separable inputs of length N.
                  The first argument of this function is a Tuple corresponding the centered indices.
+ `sz`:           The size of the N-dimensional array to create
+ `args`...:      a list of arguments, each being an N-dimensional vector
+ `all_axes`: if provided, this memory is used instead of allocating a new one. This can be useful if you want to use the same memory for multiple calculations.
+ `offset`:       position of the center from which the position is measured
+ `scale`:        defines the pixel size as vector or scalar. Default: 1.0.
+ `operator`:    the separable operator connecting the separable dimensions

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
    operator = *, all_axes = get_bc_mem(AT, sz, operator, get_arg_sz(sz, args...)),
    kwargs...) where {AT, N}
    # replace the sep memory inside the broadcast structure with new values:
    calculate_separables(AT, fct, sz, args...; all_axes = all_axes.args, kwargs...)
    return all_axes
    # Broadcast.instantiate(Broadcast.broadcasted(operator, calculate_separables(AT, fct, sz, args...; all_axes=all_axes, kwargs...)...))
end

function calculate_broadcasted(fct, sz::NTuple{N, Int}, args...;
        operator = *, all_axes = get_bc_mem(AT, sz, operator, get_arg_sz(sz, args...)),
        kwargs...) where {N}
        calculate_separables(DefaultArrType, fct, sz, args...; all_axes=all_axes.args, kwargs...)
        return all_axes
        # Broadcast.instantiate(Broadcast.broadcasted(operator, calculate_separables(DefaultArrType, fct, sz, args...; all_axes=all_axes, kwargs...)...))
end

# function calculate_sep_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, args...; 
#     all_axes = (similar_arr_type(AT, eltype(AT), Val(1)))(undef, sum(sz)),
#     operator = *, defaults = nothing, kwargs...) where {AT, N}
#     # defaults should be evaluated here and filled into args...
#     return calculate_separables_nokw(AT, fct, sz, args...; all_axes=all_axes, kwargs...)
# end

# function calculate_sep_nokw(fct, sz::NTuple{N, Int}, args...; 
#     all_axes = (similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(1)))(undef, sum(sz)),
#     operator = *, defaults = nothing, kwargs...) where {N}
#     return calculate_separables_nokw(AT, fct, sz, args...; all_axes=all_axes, kwargs...)
# end

### Versions where offst and scale are without keyword arguments
function calculate_broadcasted_nokw(::Type{AT}, fct, sz::NTuple{N, Int}, args...; 
    operator = *, all_axes = get_bc_mem(AT, sz, operator, get_arg_sz(sz, args...)),
    defaults = nothing, kwargs...) where {AT, N}
    # defaults should be evaluated here and filled into args...
    calculate_separables_nokw_hook(AT, fct, sz, args...; all_axes=all_axes.args, kwargs...)
    # res = Broadcast.instantiate(Broadcast.broadcasted(operator, calculate_separables_nokw_hook(AT, fct, sz, args...; all_axes=all_axes, kwargs...)...))
    # @show eltype(collect(res))
    return all_axes
end

function calculate_separables_nokw_hook(::Type{AT}, fct, sz::NTuple{N, Int}, args...; kwargs...) where {AT, N} 
    return calculate_separables_nokw(AT, fct, sz, args...; kwargs...)
end

# just a dummy to divert the function definition below.
function calculate_broadcasted_nokw_xxx()
end

# this code only works for the multiplicative version and actually only saves very few allocations.
# it's not worth the specialization:
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(calculate_broadcasted_nokw), ::Type{AT}, fct, sz::NTuple{N, Int}, args...;
    operator = *, all_axes = get_bc_mem(AT, sz, operator, get_arg_sz(sz, args...)),
    defaults = nothing, kwargs...) where {AT, N}

    # println("in rrule broadcast")
    # y = calculate_broadcasted_nokw(AT, fct, sz, args..., kwargs...)
    y_sep, calculate_sep_nokw_pullback = rrule_via_ad(config, calculate_separables_nokw_hook, AT, fct, sz, args...; operator = operator, all_axes=all_axes.args, kwargs...)
    # println("y done")
    # y = Broadcast.instantiate(Broadcast.broadcasted(operator, y_sep...))
    y = operator.(y_sep...) # is faster than the broadcast above.
    # @show size(y_sep[1])
    # @show size(y_sep[2])
    # @show size(y)

    function calculate_broadcasted_nokw_pullback(dy)
        # println("in debug_dummy") # sz is (10, 20)
        # @show size(dy)
        projected = ntuple((d) -> begin
            dims=((1:d-1)...,(d+1:N)...)
            reduce(+, operator.(y_sep[[dims...]]...) .* dy, dims=dims)
            end, Val(N)) # 7 Mb
        # @show typeof(projected[1])
        # @show size(projected[1])
        # @show typeof(projected[2])
        myres = calculate_sep_nokw_pullback(projected) # 8 kB
        return myres
    end
    return y, calculate_broadcasted_nokw_pullback # in_place_assing_pullback # in_place_assing_pullback
end

# function calculate_separables_nokw_hook2()
# end

# Needs revision
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(calculate_separables_nokw_hook), ::Type{AT}, fct, sz::NTuple{N, Int}, args...;
    all_axes = get_sep_mem(AT, sz, get_arg_sz(sz, args...)), kwargs...) where {AT, N}
    # ids = ntuple((d) -> reorient(get_1d_ids(d, sz, args[1], args[2]), d, Val(N)), Val(N)) # offset==args[1] and scale==args[2]
    ids = ntuple((d) -> get_1d_ids(d, sz, args[1], args[2]), Val(N)) # offset==args[1] and scale==args[2]
    ids_offset_only = get_1d_ids.(1:N, Ref(sz), Ref(args[1]), one(eltype(AT)))

    y = calculate_separables_nokw(AT, fct, sz, args...; all_axes = all_axes, kwargs...)

    args_1d = ntuple((d)-> pick_n.(d, args[3:end]), Val(N))

    function calculate_separables_nokw_hook_pullback(dy)
        #get_idx_gradient(fct, length(sz), y, x, sz[d], dy[d], args...)
        #get_arg_gradient(fct, length(sz), y, x, sz[d], dy[d].*ids_offset_only[d], args...)
        yv = ntuple((d) -> (@view y[d][:]), Val(N))
        dyv = ntuple((d) -> (@view dy[d][:]), Val(N))
        doffset = optional_convert(args[1], ntuple((d) -> .- pick_n(d, args[2]) .* 
            get_idx_gradient(fct, length(sz), yv[d], ids[d], sz[d], dyv[d], args_1d[d]...), Val(N))) # ids @ offset the -1 is since the argument of fct is idx-offset

        dscale = optional_convert(args[2], ntuple((d) -> 
            get_idx_gradient(fct, length(sz), yv[d], ids[d], sz[d], dyv[d].* ids_offset_only[d], args_1d[d]...), Val(N))) # ids @ offset the -1 is since the argument of fct is idx-offset        

        dargs = ntuple((argno) -> optional_convert(args[2+argno],
            ntuple((d) -> get_arg_gradient(fct, length(sz), yv[d], ids[d], sz[d], dyv[d], args_1d[d]...), Val(N))), length(args)-2) # ids @ offset the -1 is since the argument of fct is idx-offset


        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), doffset, dscale, dargs...)
        end
    return y, calculate_separables_nokw_hook_pullback
end


function get_fg!(data::AbstractArray{T,N}, fct, prod_dims=N; loss = loss_gaussian, bg=zero(real(T))) where {T,N}
    RT = real(eltype(data))
    AT = typeof(data)
    RAT = similar_arr_type(AT, RT, Val(1))

    operator = get_operator(fct)
    sz = size(data)[1:prod_dims]
    hyper_sz = 1;
    if prod_dims < N
        hyper_sz = size(data)[prod_dims+1:end]
    end
    by = get_bc_mem(AT, size(data)[1:prod_dims], operator, hyper_sz)
    y = by.args
    # yv = ntuple((d) -> (@view y[d]), Val(prod_dims))

    dy = get_sep_mem(typeof(data), size(data)[1:prod_dims], hyper_sz)
    # dyv = ntuple((d) -> (@view dy[d]), Val(prod_dims))

    resid = similar(data) # checkpoint

    loss_fg!, C = get_fgC!(data, loss, bg)

    # this function returns the forward value and mutates the gradient G
    function fg!(F, G, vec)
        bg = hasproperty(vec, :bg) ? vec.bg : RAT([zero(RT)]);
        intensity = hasproperty(vec, :intensity) ? vec.intensity : RAT([one(RT)])
        off = hasproperty(vec, :off) ? vec.off : (sz.÷2 .+1)
        sca = hasproperty(vec, :sca) ? vec.sca : RAT([one(RT)])
        args = hasproperty(vec, :args) ? (vec.args,) : Tuple([])
        # mid = sz .÷ 2 .+ 1
        # off = off .+ mid
        ids = ntuple((d) -> get_1d_ids(d, sz, off, sca), Val(prod_dims)) # offset==args[1] and scale==args[2]
        ids_offset_only = get_1d_ids.(1:prod_dims, Ref(sz), Ref(off))

        # 5kB, result is in by
        calculate_broadcasted_nokw(typeof(data), fct, sz, off, sca, args...; operator=operator, all_axes=by)

        # if !isnothing(F) || !isnothing(G)
        #     resid .= bg .+ intensity .* by .- data
        # end
        loss = loss_fg!(F, resid, bg .+ intensity .* by)
        if !isnothing(G)
            # for arrays this should be .=
            G.bg = C*sum(resid)

            other_dims = ntuple((d)-> (ntuple((n)->n, d-1)..., ntuple((n)->d+n, prod_dims-d)...), Val(prod_dims))
            # @show other_dims
            other_ys = ntuple((d)-> (ntuple((n)->y[n], d-1)..., ntuple((n)->y[d+n], prod_dims-d)...), Val(prod_dims))
            # moved 2*intensity to the condensed terms, but logically it should be in dy!
            # this is fairly expensive in memory:
            # dy = ntuple((d) -> sum(resid.* (.*(other_ys[d]...)), dims=other_dims[d]), Val(N))

            for d in 1:prod_dims
                # This costs 33 kB allocation, sometimes it can be faster?
                # dy[d] .= sum(resid.* (.*(other_ys[d]...)), dims=other_dims[d])
                # this costs 2 kB allocations but is slower
                dy[d] .= broadcast_reduce(*, +, resid, other_ys[d]..., dims=other_dims[d], init=zero(T))
            end
            # less memory, but a little slower:
            # dy = ntuple((d) -> broadcast_reduce(*, +, resid, other_ys[d]..., dims=other_dims[d], init=zero(T)), Val(N))
            # @time dy = ntuple((d) -> mapreduce(*, +, resid, other_ys[d]...), dims=other_dims[d]), Val(N))

            # args_1d = ntuple((d)-> pick_n.(d, args), Val(prod_dims))
            args_1d = ntuple((d)-> get_vec_dim.(args, d, Ref(sz)), Val(prod_dims))
            if hasproperty(vec, :off)                
                G.off .= optional_convert(off, 
                    ntuple((d) -> (-C*get_vec_dim(intensity,d, sz) .* get_vec_dim(sca, d, sz)) .* 
                    get_idx_gradient(fct, length(sz), y[d], ids[d], sz[d], dy[d], args_1d[d]...), Val(prod_dims))) # ids @ offset the -1 is since the argument of fct is idx-offset
            end

            if hasproperty(vec, :args)
                red_dims = (isa(args_1d[1][1], AbstractVector) || isa(args_1d[1][1], Number)) ? length(size(data)) : length(sz)
                tg = ntuple((d) -> C * get_vec_dim(intensity, d, sz) .* get_arg_gradient(fct, red_dims, y[d], ids[d], sz[d], dy[d], args_1d[d]...), Val(prod_dims))
                dargs = ntuple((argno) -> optional_convert(args[argno], tg), length(args)) # ids @ offset the -1 is since the argument of fct is idx-offset
                G.args .= dargs[1]
            end
            # dargs = (0f0,0f0)

            if hasproperty(vec, :sca)
                # missuse the dy memory
                for d = 1:prod_dims
                    dy[d] .= dy[d].* ids_offset_only[d]
                end
                G.sca .= optional_convert(sca, ntuple((d) -> 
                C * get_vec_dim(intensity, d, sz) .* get_idx_gradient(fct, length(sz), y[d], ids[d], sz[d], dy[d], args_1d[d]...), Val(prod_dims))) # ids @ offset the -1 is since the argument of fct is idx-offset
            end
                # 1.5 kB:
            # (2*intensity).*get_idx_gradient(fct, length(sz), yv[d], ids[d], sz[d], dyv[d].* ids_offset_only[d], args_1d[d]...), Val(N))) # ids @ offset the -1 is since the argument of fct is idx-offset
            # dscale = (0f0,0f0)
        end

        # return loss, dfdbg, dfdI, doffset, dscale, dargs...
        # loss = (!isnothing(F)) ? mapreduce(abs2, +, resid; init=zero(T)) : T(0);

        if !isnothing(G) # forward needs to be calculated
            # resid .*= by # is slower!
            resid .= resid .* by
            G.intensity = C*sum(resid)
            # G.intensity = 2*sum(resid.*by)
        end

        return loss
        end
    return fg!
end

function calculate_broadcasted_nokw(fct, sz::NTuple{N, Int}, args...; 
    operator = *, all_axes = get_bc_mem(AT, sz, operator),
    defaults = nothing, kwargs...) where {N}
    # defaults should be evaluated here and filled into args...
    calculate_separables_nokw(DefaultArrType, fct, sz, args...; all_axes=all_axes.args, kwargs...)
    # res = Broadcast.instantiate(Broadcast.broadcasted(operator, calculate_separables_nokw(DefaultArrType, fct, sz, args...; all_axes=all_axes, kwargs...)...))
    # @show eltype(collect(res))
    return all_axes
end

# towards a Gaussian that can also be rotated:
# mulitply with exp(-(x-x0)*(y-y0)/(sigma_xy)) = exp(-(x-x0)/(sigma_xy)) ^ (y-y0) 
# g = gaussian_sep((200, 200), sigma=2.2)
# vg = Base.broadcasted((x,y)->x, collect(g), ones(1,1,10));
# res = similar(g.args[1], size(vg));
# @time gg = accumulate!(*, res, vg, dims=3);  #, init=collect(g)


"""
    separable_view{N}(fct, sz, args...; offset =  sz.÷2 .+1, scale = one(real(eltype(AT))), operator = .*)

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
+ `operator`:    the separable operator connecting the separable dimensions

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
        operator = *, kwargs...) where {AT, N}
    res = calculate_separables(AT, fct, sz, args...; operator=operator, kwargs...)
    return LazyArray(@~ operator.(res...)) # to prevent premature evaluation
end

function separable_view(fct, sz::NTuple{N, Int}, args...; 
        operator = *, kwargs...) where {N}
    separable_view(DefaultArrType, fct, sz::NTuple{N, Int}, args...; operator=operator, kwargs...)
end

"""
    separable_create([::Type{TA},] fct, sz::NTuple{N, Int}, args...; offset =  sz.÷2 .+1, scale = one(real(eltype(AT))), operator = *, kwargs...) where {TA, N}

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
+ `operator`:    the separable operator connecting the separable dimensions

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
function separable_create(::Type{TA}, fct, sz::NTuple{N, Int}, args...; operator::Function = *, kwargs...)::similar_arr_type(TA, T, Val(N)) where {T, N, TA <: AbstractArray{T}}
    # res = calculate_separables(TA, fct, sz, args...; kwargs...)
    # operator.(res...)
    res = similar(TA, sz)
    res .= calculate_broadcasted(TA, fct, sz, args...; operator=operator, kwargs...)
    return res
end

## the code below seems not type-stable but the code above is. Why?
function separable_create(fct, sz::NTuple{N, Int}, args...; operator::Function = *, kwargs...)::similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(N)) where {N}
    # TT = similar_arr_type(DefaultArrType, Float32, Val(N))
    # res = calculate_separables(similar_arr_type(DefaultArrType, eltype(DefaultArrType), Val(N)), fct, sz, args...; kwargs...)
    # return operator.(res...)
    # resT = 
    separable_create(similar_arr_type(DefaultArrType, Float32, Val(N)), fct, sz, args...; operator=operator, kwargs...)
end
