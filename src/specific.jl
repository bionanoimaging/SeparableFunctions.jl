# function gaussian_sep(::Type{TA}, sz::NTuple{N, Int}, sigma,; offset=sz.÷2 .+1) where {TA, N}
#     invsigma22 = 1 ./(2 .*sigma.^2)
#     fct = (r, invsigma22, pos) -> exp(-(r-pos)^2*invsigma22)
#     separable_create(TA, fct, sz, invsigma22; offset=offset)
# end

# function gaussian_sep(sz::NTuple{N, Int}, sigma, ; offset=sz.÷2 .+1) where {N}
#     gaussian_sep(DefaultArrType, sz, sigma; offset=offset)
# end

# function gaussian_sep_lz(::Type{TA}, sz::NTuple{N, Int}, sigma, ; offset=sz.÷2 .+1) where {TA, N}
#     invsigma22 = 1 ./(2 .*sigma.^2)
#     fct = (r, invsigma22, pos) -> exp(-(r-pos)^2*invsigma22)
#     separable_view(TA, fct, sz, invsigma22; offset=offset)
# end

# function gaussian_sep_lz(sz::NTuple{N, Int}, sigma, ; offset=sz.÷2 .+1) where {N}
#     gaussian_sep_lz(DefaultArrType, sz, sigma; offset=offset)
# end
using ChainRulesCore

export mem_fct, raw_fct

function generate_functions_expr()
    # offset and scale is already wrapped in the generator function
    # x_expr = :(scale .* (x .- offset))

    functions = [
        # Note that there is a problem with the CUDA toolbox. It does not support kwargs (in broadcasting).
        # therefore this toolbox uses a mechanism to convert kwargs into normal args. It is a bit of a hack.
        # A limitation is that only the last argument can use "nothing", get ignored, if not provided, and specify a sz-dependent calculation in the actual function header.
        # This means that this argument can alternatively be supplied as a non-named argument and it will still work.
        # Rules: the calculation function has no kwargs but the last N arguments are the kwargs of the wrapper function
        # FunctionName, kwarg_names, no_kwargs_function_definition, default_return_type, default_separamble_operator
        (:(gaussian),(sigma=1.0,), :((x,sz, sigma) -> exp(-x^2/(2*sigma^2))), Float32, *, 
            :((f, x, sz, sigma) -> -x/sigma^2 * f),
            :((f, x, sz, sigma) -> x^2 /sigma^3 * f)
            ), 
        (:(normal), (sigma=1.0,), :((x,sz, sigma) -> exp(- x^2/(2*sigma^2))/(sqrt(eltype(x)(2pi))*abs(sigma))), Float32, *,
            :((f, x, sz, sigma) -> -x/sigma^2 * f),
            :((f, x, sz, sigma) -> (x^2 /sigma^3 - inv(sigma)) * f)
            ),
        (:(sinc), NamedTuple(), :((x,sz) -> sinc(x)), Float32, *,
            :((f, x, sz) -> ifelse(x == zero(eltype(x)), zeros(eltype(x), size(x)), (cospi(x) - f)/x))
            ),
        # the value "nothing" means that this default argument will not be handed over. But this works only for the last argument!
        (:(exp_ikx), (shift_by=nothing,), :((x,sz, shift_by=sz÷2) -> cis(x*(-eltype(x)(2pi)*shift_by/sz))), ComplexF32, *,
            :((f, x, sz, shift_by) -> (-1im*eltype(x)(2pi)*shift_by/sz) * f),
            :((f, x, sz, shift_by) -> (-1im*eltype(x)(2pi)/sz) *x * f)
            ),
        (:(ramp), (slope=0,), :((x,sz, slope) -> slope*x), Float32, +,
            :((f, x, sz, slope) ->  slope),
            :((f, x, sz, slope) ->  x)
            ), # different meaning than IFA ramp
        (:(rr2), NamedTuple(), :((x, sz) -> (x*x)), Float32, +,
            :((f, x, sz) ->  2 * x)
            ),
        (:(box), (boxsize=nothing,), :((x, sz, boxsize=sz/2) -> abs(x) <= (boxsize/2)), Bool, *,
            :((f, x, sz) -> one(eltype(x)))
            ),
    ]
    return functions
end

# would be nice to have a macro which defines all those function extensions.
# But its not quite that simple. First try:
# macro define_separable(basename, fct, basetype, operator)
#     @eval function $(Symbol(basename, :_col))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
#         fct = $(fct) # to assign the function to a symbol
#         separable_create(TA, fct, sz, args...; operator=$(operator), kwargs...)
#     end
# end

for F in generate_functions_expr() 
    # default functions with offset and scaling behavior
 
    # define the _raw function
    @eval function $(Symbol(F[1], :_raw))(x, sz, args...)
        return $(F[3])(x, sz, args...) 
    end
    # just the raw version of the function
    @eval export $(Symbol(F[1], :_raw))

    if (length(F) == 6) # a gradient definition was provided explicitely
        # @show "creating rrule for $(Symbol(F[1], :_raw)) 
        @eval function get_idx_gradient(::typeof($(Symbol(F[1], :_raw))), prod_dims, y, x, sz, dy)
            # println("in set_idx_gradient")
            return mapreduce(*, +, conj.(dy), $(F[6]).(y, x, sz); dims=1:prod_dims)
        end

        @eval function ChainRulesCore.rrule(::typeof($(Symbol(F[1], :_raw))), x, sz; kwargs...) 
            # println("in rrule raw")
            y = $(Symbol(F[1], :_raw))(x, sz; kwargs...) # to assign the function to a symbol
            function mypullback(dy)
                mydx =  conj.(dy) .* $(F[6])(y, x, sz; kwargs...)
                return NoTangent(), mydx, NoTangent()
            end
            return y, mypullback
        end
        # @show "added rrule for $(Symbol(F[1], :_raw))"
    end
    if (length(F) == 7) # a gradient definition was provided explicitely
        # @show "creating rrule for $(Symbol(F[1], :_raw)) "
        @eval function get_idx_gradient(::typeof($(Symbol(F[1], :_raw))), prod_dims, y, x, sz, dy, args...)
            # println("in set_idx_gradient")
                return mapreduce(*, +, conj.(dy), $(F[6]).(y, x, sz, args...); dims=1:prod_dims)
        end

        @eval function get_arg_gradient(::typeof($(Symbol(F[1], :_raw))), prod_dims, y, x, sz, dy, args...)
            # println("in set_arg_gradient")
            return mapreduce(*, +, conj.(dy), $(F[7]).(y, x, sz, args...), dims=1:prod_dims) 
        end

        @eval function ChainRulesCore.rrule(::typeof($(Symbol(F[1], :_raw))), x, sz, args...; kwargs...) 
            # println("in rrule2 raw")
            y = $(Symbol(F[1], :_raw))(x, sz, args...; kwargs...) # to assign the function to a symbol
            function mypullback(dy)
                # println("pb")
                # @show dy
                # @show $(F[6])(y, x, sz, args...; kwargs...)
                mydx =  conj.(dy) .* $(F[6])(y, x, sz, args...; kwargs...)
                # targ = ntuple(d -> begin
                #     mydarg = F[6+d]
                #     dy .* $(mydarg)(y, x, sz, args...; kwargs...)
                #     end, length(args))
                # @show size($(F[7])(y, x, sz, args...; kwargs...))
                # @show dy
                # @show dy .* $(F[7])(y, x, sz, args...; kwargs...)
                mydarg = dot(dy, $(F[7])(y, x, sz, args...; kwargs...)) 
                # mydarg = sum(dy .* $(F[7])(y, x, sz, args...; kwargs...)) 
                return NoTangent(), mydx, NoTangent(), mydarg
            end
            return y, mypullback
        end
        # @show "added rrule for $(Symbol(F[1], :_raw))"
    end
    @eval function get_operator(::typeof($(Symbol(F[1], :_raw)))) 
        return $(F[5])
    end

    @eval function $(Symbol(F[1], :_col))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_create(TA, fct, sz, args...; defaults=$(F[2]), operator=$(F[5]), kwargs...)
    end
 
    @eval function $(Symbol(F[1], :_col))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_create(Array{$(F[4])}, fct, sz, args...; defaults=$(F[2]), operator=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_sep))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        calculate_broadcasted(TA, fct, sz, args...; defaults=$(F[2]), operator=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_sep))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        calculate_broadcasted(Array{$(F[4])}, fct, sz, args...; defaults=$(F[2]), operator=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_nokw_sep))(::Type{TA}, sz::NTuple{N, Int}, args...;
                        all_axes = get_bc_mem(TA, sz, $(F[5]), get_arg_sz(sz, args...))
                    ) where {TA, N}
        # fct = $(F[3]) # to assign the function to a symbol

        # @show "call"
        return calculate_broadcasted_nokw(TA, $(Symbol(F[1], :_raw)), sz, args...; defaults=$(F[2]), operator=$(F[5]),
            all_axes=all_axes)
        # operator=$(F[5])
        # return calculate_separables_nokw(TA, fct, sz, args...; all_axes=all_axes), operator
    end

    @eval function $(Symbol(F[1], :_nokw_sep))(sz::NTuple{N, Int}, args...;
                        all_axes = get_bc_mem(Array{$(F[4])}, sz, $(F[5]), get_arg_sz(sz, args...))
                    ) where {N}
        # fct = $(F[3]) # to assign the function to a symbol        
        # @show "call2" 
        # @show typeof(Array{$(F[4])})
        # @show typeof($(F[4]))
        # @show sz
        # @show args
        # @show asz = get_arg_sz(sz, args...)
        ##### Zygote messes this up!:
        # @show typeof(get_bc_mem(Array{$(F[4])}, sz, $(F[5]), asz))
        # @show typeof(all_axes)
        return calculate_broadcasted_nokw(Array{$(F[4])}, $(Symbol(F[1], :_raw)), sz, args...; defaults=$(F[2]), operator=$(F[5]), all_axes=all_axes)
        # operator=$(F[5])
        # return calculate_separables_nokw(Array{$(F[4])}, fct, sz, args...; all_axes=all_axes), operator
    end

    @eval function $(Symbol(F[1], :_vec))(::Type{TA}, sz::NTuple{N, Int}, vec;
        all_axes = nothing) where {TA, N}
        RT = real(eltype(TA))
        intensity = (hasproperty(vec, :intensity)) ? get_vec_dim(vec.intensity, 1, sz) : one(RT)
        bg = (hasproperty(vec, :bg)) ? get_vec_dim(vec.bg, 1, sz) : nothing

        off = (hasproperty(vec, :off)) ? vec.off : nothing
        sca = (hasproperty(vec, :sca)) ? vec.sca : nothing
        args = (hasproperty(vec, :args)) ? (vec.args,) : ()
        if any(isa.(args, Tuple))
            error("use vectors rather than tuples in component arrays, since Zygote has trouble with tuples.")
        end
        all_axes = isnothing(all_axes) ? get_bc_mem(TA, sz, $(F[5]), get_arg_sz(sz, off, sca, bg, intensity, args...)) : all_axes;        
        # use the return value instead of all_axes directly, since only this triggers the gradient calculation correctly
        return bg .+ intensity .* ($(Symbol(F[1], :_nokw_sep))(TA, sz, off, sca, args...; all_axes=all_axes))
    end

    @eval function $(Symbol(F[1], :_vec))(sz::NTuple{N, Int}, vec;
        all_axes = nothing) where {N}
        T = $(F[4])
        TA = Array{T}
        if hasproperty(vec, :off) && isa(vec.off, AbstractArray)
            T = promote_type(T, eltype(vec.off))
            TA = similar_arr_type(typeof(vec.off), T, Val(N))
        end
        if hasproperty(vec, :intensity) && isa(vec.intensity, AbstractArray)
            T = promote_type(T, eltype(vec.intensity))
            TA = similar_arr_type(typeof(vec.intensity), T, Val(N))
        end
        if hasproperty(vec, :sca) && isa(vec.sca, AbstractArray)
            T = promote_type(T, eltype(vec.sca))
            TA = similar_arr_type(typeof(vec.sca), T, Val(N))
        end
        if hasproperty(vec, :args) && isa(vec.args, AbstractArray)
            T = promote_type(T, eltype(vec.args))
            TA = similar_arr_type(typeof(vec.args), T, Val(N))
        end
        return $(Symbol(F[1], :_vec))(TA, sz, vec; all_axes=all_axes)        
    end

    @eval function $(Symbol(F[1], :_lz))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_view(TA, fct, sz, args...; defaults=$(F[2]), operator=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_lz))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_view(Array{$(F[4])}, fct, sz, args...; defaults=$(F[2]), operator=$(F[5]), kwargs...)
    end

    @eval mem_fct(::typeof($(Symbol(F[1], :_col))), ::Type{AT}, sz, hyper_sz=()) where AT = get_bc_mem(AT, sz, $(F[5]), hyper_sz)
    @eval mem_fct(::typeof($(Symbol(F[1], :_sep))), ::Type{AT}, sz, hyper_sz=()) where AT = get_bc_mem(AT, sz, $(F[5]), hyper_sz)
    @eval mem_fct(::typeof($(Symbol(F[1], :_nokw_sep))), ::Type{AT}, sz, hyper_sz=()) where AT = get_bc_mem(AT, sz, $(F[5]), hyper_sz)
    @eval mem_fct(::typeof($(Symbol(F[1], :_vec))), ::Type{AT}, sz, hyper_sz=()) where AT = get_bc_mem(AT, sz, $(F[5]), hyper_sz)
    @eval mem_fct(::typeof($(Symbol(F[1], :_lz))), ::Type{AT}, sz, hyper_sz=()) where AT = get_sep_mem(AT, sz, hyper_sz)

    @eval raw_fct(::typeof($(Symbol(F[1], :_raw)))) = $(Symbol(F[1], :_raw))
    @eval raw_fct(::typeof($(Symbol(F[1], :_col)))) = $(Symbol(F[1], :_raw))
    @eval raw_fct(::typeof($(Symbol(F[1], :_sep)))) = $(Symbol(F[1], :_raw))
    @eval raw_fct(::typeof($(Symbol(F[1], :_nokw_sep)))) = $(Symbol(F[1], :_raw))
    @eval raw_fct(::typeof($(Symbol(F[1], :_vec)))) = $(Symbol(F[1], :_raw))
    @eval raw_fct(::typeof($(Symbol(F[1], :_lz)))) = $(Symbol(F[1], :_raw))
    # collected: fast separable calculation but resulting in an ND array
    @eval export $(Symbol(F[1], :_col))
    # separated: a vector of separated contributions is returned and the user has to combine them
    @eval export $(Symbol(F[1], :_sep))
    # a broadcasted version which accepts a ComponentArray as an input
    @eval export $(Symbol(F[1], :_vec))
    # lazy: A LazyArray representation is returned
    @eval export $(Symbol(F[1], :_nokw_sep))
    # @eval export $(Symbol(F[1], :_lz))
end 

## Here some individual versions based on copy_corners! stuff. They only exist in the _cor version as they are not separable in X and Y.
"""
    propagator_col([]::Type{TA},] sz::NTuple{N, Int}; Δz=one(eltype(TA)), k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1)), ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=use_sep) where{TA, N}

generates a propagator for propagating optical fields via exp(i kz Δz) with kz=sqrt(k0^2-kx^2-ky^2). The k-space radius is stated by
k_max relative to the Nyquist frequency, as long as the scale remains to be 1 ./ (2 max.(sz ./ 2, 1))).

If the array has 3 dimensions, a stack of equal-distance propagators will be generated with the slice
sz[3]÷2+1  corresponding to the mid position yielding no phase change.

Note that there is no `propagator_sep` version of this function, since this propagator is not fully separable. 

#Arguments
+ `TA`:     type of the array to generate. E.g. Array{Float64} or CuArray{Float32}.
+ `sz`:     size of the array to generate.  If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `Δz`:     distance in Z to propagate per slice.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate. By default, the (Fourier space) center position along Z is chosen.
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.

"""
function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=one(eltype(TA)), k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1)), ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=false) where{TA, N}
# function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1))) where{TA, N}
    if length(sz) > 3
        error("propagators are only allowed up to the third dimension. If you need to propagate several stacks, use broadcasting.")
    end
    arr = TA(undef, sz)
    propagator_col!(arr; Δz=Δz, k_max=k_max, scale=scale, ref_idx = ref_idx, use_sep=use_sep) 
end

function propagator_col(sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1)), ref_idx =  (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=false) where{N}
    propagator_col(DefaultComplexArrType, sz; Δz=Δz, k_max=k_max, scale=scale, ref_idx = ref_idx, use_sep=use_sep)
end

"""
    propagator_col!(arr::AbstractArray{T,N}; Δz=one(eltype(TA)), ref_idx = size(arr,3)÷2+1, k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1)), use_sep=false) where{TA, N}

generates a propagator for propagating optical fields via exp(i kz Δz) with kz=sqrt(k0^2-kx^2-ky^2). The k-space radius is stated by
k_max relative to the Nyquist frequency, as long as the scale remains to be 1 ./ (2 max.(sz ./ 2, 1))).

If `arr` has 3 dimensions, a stack of equal-distance propagators will be generated with the slice
size(arr,3)÷2+1  corresponding to the mid position yielding no phase change.

Note that there is no `propagator_sep` version of this function, since this propagator is not fully separable. 

# Arguments
+ `arr`:    the array to fill with propagators. If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `Δz`:     distance in Z to propagate per slice in relation to the wavelength. Nyquist sampling would be 0.5.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere in relation to sampling frequency. This is not the aperture limit!
            k_max = 0.5 corresponds to the Nyquist limit.
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate. By default, the (Fourier space) center position along Z is chosen.
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.

# Example
```julia
julia> λ = 0.500 # wavelength (e.g. in µm)
julia> sampling = (0.25, 0.25, 0.25) # in the same units
julia> tmp = zeros(ComplexF32, 100, 50, 30) # will be overwritten below
julia> # here is an example of sampling the amplitude (not intensity!) just at the Nyquist limit:
julia> p = propagator_col!(tmp, Δz = sampling[3] / λ, k_max = sampling[1:2] ./ λ)
julia> # this is equivalent to:
julia> p = propagator_col!(tmp, Δz = 0.5)
julia> # for convenience, there is also a version accepting the parameters λ (in the medium) and sampling.
julia> # And we can define the z-index at which no propagation is obtained:
julia> p = propagator_col!(tmp, sampling, λ; ref_idx = 1)
julia> # Note that a 2D propagator is always propagating by one Δz, thus corresponding to slice 2 in the above result:
julia> tmp2d = zeros(ComplexF32, 100, 50);
julia> p = propagator_col!(tmp2d, sampling, λ)
```
"""    
function propagator_col!(arr::AbstractArray{T,N}; Δz=one(eltype(arr)), ref_idx = size(arr,3)÷2+1, k_max=0.5f0, scale=0.5f0 ./ (max.(size(arr) ./ 2, 1)), use_sep=false) where{T, N}
    # function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1))) where{TA, N}
    sz = size(arr)
    RT = real(eltype(arr))
    if isa(k_max, NTuple{2})
        scale = scale[1:2] .* RT(0.5) ./ k_max[1:2];
        k_max = RT(0.5)
    end

    k2_max = RT.(k_max .^2)

    # fac = eltype(arr)(4im * pi * Δz)
    # f(r2) = cispi(sqrt(max(zero(real(eltype(TA))),k2_max - r2)) * (4 * Δz))
    # f(r2) = exp(sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac)
    fac = RT(4pi * Δz) # Due to scale being 0.5 this factor is 4pi instead of 2pi
    if length(sz) < 3 || sz[3] == 1
        # f2d(r2) = cis(sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac)
        # return f2d.(rr2sep, k2_max); 
        if (use_sep)
            # For some reason the function f2d above is much slower in CUDA than the line below
            rr2sep = rr2_sep(real_arr_type(typeof(arr)), sz[1:2]; scale = scale) 
            f2ds(r2) = (r2 >= k2_max) ? one(eltype(arr)) : cis(fac * sqrt(k2_max - r2))
            arr .= f2ds.(rr2sep)
            return arr
        else
            # f2d(r2) = cis(sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac)
            f2dr(r2) = (r2 >= k2_max) ? one(eltype(arr)) : cis(fac * sqrt(k2_max - r2))
            # f2d(r2) = (r2 > k2_max) ? one(eltype(arr)) : cis(sqrt(k2_max - r2) * fac);
            # f2d(r2) = cis(sqrt(r2))
            return calc_radial2_symm!(arr, f2dr; scale=scale); 
        end
    else
        if (ref_idx < sz[3])
            # ifelse cannot be used due to the second expressing being evaluated anyway
            f(r2) = (r2 >= k2_max) ? one(eltype(arr)) : cis(fac * sqrt(k2_max - r2))
            # f(r2) = ifelse(r2 >= k2_max, one(eltype(arr)), cis(sqrt(k2_max - r2) * fac))
            zref = ref_idx + 1; # The plane which contains one single propagation operation
            ref_slice = @view arr[:,:,zref];
            if (use_sep)
                rr2sep = rr2_sep(real_arr_type(typeof(arr)), sz[1:2]; scale = scale) 
                ref_slice .= f.(rr2sep)
            else
                calc_radial2_symm!(ref_slice, f; scale=scale); 
            end
            for z=zref+1:sz[3]
                arr[:,:,z] .= (@view arr[:,:,z-1]) .* ref_slice;
            end
            for z=zref-1:-1:1
                arr[:,:,z] .= (@view arr[:,:,z+1]) .* conj.(ref_slice);
            end
            return arr
        else # if the reference is the last slice: use a different algorithm going backwards
            g(r2) = (r2 >= k2_max) ? one(eltype(arr)) : cis(-fac * sqrt(k2_max - r2))
            zref = ref_idx - 1; # The plane which contains one single propagation operation
            ref_slice = @view arr[:,:,zref];
            if (use_sep)
                rr2sep = rr2_sep(real_arr_type(typeof(arr)), sz[1:2]; scale = scale) 
                ref_slice .= g.(rr2sep)
            else
                calc_radial2_symm!(ref_slice, g; scale=scale); 
            end
            for z=zref+1:sz[3]
                arr[:,:,z] .= (@view arr[:,:,z-1]) .* conj.(ref_slice);
            end
            for z=zref-1:-1:1
                arr[:,:,z] .= (@view arr[:,:,z+1]) .* ref_slice;
            end
            return arr
        end
    end
end

"""
    propagator_col!(arr::AbstractArray{T,N}, sampling::NTuple{3}, λ; ref_idx = size(arr,3)÷2+1) where{T, N}

generates a propagator for propagating optical fields via exp(i kz Δz) with kz=sqrt(k0^2-kx^2-ky^2). The k-space radius is stated by
k_max relative to the Nyquist frequency, as long as the scale remains to be 1 ./ (2 max.(sz ./ 2, 1))).

If `arr` has 3 dimensions, a stack of equal-distance propagators will be generated with the slice
size(arr,3)÷2+1  corresponding to the mid position yielding no phase change.

Note that there is no `propagator_sep` version of this function, since this propagator is not fully separable. 

# Arguments
+ `arr`:    the array to fill with propagators. If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `sampling`: pixelsize (e.g. in micrometer)
+ `λ`:  wavelength (same units, i.e. micrometer).
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate. By default, the (Fourier space) center position along Z is chosen.
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.

# Example
```julia
julia> λ = 0.500 # wavelength (e.g. in µm)
julia> sampling = (0.25, 0.25, 0.25) # in the same units
julia> # Note that a 2D propagator is always propagating by one Δz, thus corresponding to slice 2 in the above result:
julia> tmp2d = zeros(ComplexF32, 100, 50);
julia> p = propagator_col!(tmp2d, sampling, λ)
```
"""    
function propagator_col!(arr::AbstractArray{T,N}, sampling::NTuple{3}, λ; ref_idx = size(arr,3)÷2+1, use_sep=false) where{T, N}
    return propagator_col!(arr; Δz = sampling[3] ./ λ, k_max = sampling[1:2] ./ λ, ref_idx=ref_idx, use_sep=use_sep)
end

"""
    propagator_col(sz::NTuple, sampling::NTuple{3}, λ; ref_idx = size(arr,3)÷2+1, use_sep=false) where{T, N}

generates a propagator for propagating optical fields via exp(i kz Δz) with kz=sqrt(k0^2-kx^2-ky^2). The k-space radius is stated by
k_max relative to the Nyquist frequency, as long as the scale remains to be 1 ./ (2 max.(sz ./ 2, 1))).

If `arr` has 3 dimensions, a stack of equal-distance propagators will be generated with the slice
size(arr,3)÷2+1  corresponding to the mid position yielding no phase change.

Note that there is no `propagator_sep` version of this function, since this propagator is not fully separable. 

# Arguments
+ `sz`:  the size-tuple of the array to fill with propagators. If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `sampling`: pixelsize (e.g. in micrometer)
+ `λ`:  wavelength (same units, i.e. micrometer).
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate. By default, the (Fourier space) center position along Z is chosen.
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.

# Example
```julia
julia> λ = 0.500 # wavelength (e.g. in µm)
julia> sampling = (0.25, 0.25, 0.25) # in the same units
julia> # Note that a 2D propagator is always propagating by one Δz, thus corresponding to slice 2 in the above result:
julia> p = propagator_col((100,50,30), sampling, λ)
```
"""    
function propagator_col(::Type{TA}, sz::NTuple{N, Int}, sampling::NTuple{3}, λ; ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=false) where{TA, N}
    if length(sz) > 3
        error("propagators are only allowed up to the third dimension. If you need to propagate several stacks, use broadcasting.")
    end
    arr = TA(undef, sz)
    propagator_col!(arr, sampling, λ; ref_idx=ref_idx, use_sep=use_sep)
end

function propagator_col(sz::NTuple{N, Int}, sampling::NTuple{3}, λ; ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=false) where{N}
    propagator_col(DefaultComplexArrType, sz, sampling, λ; ref_idx = ref_idx, use_sep=use_sep)
end

"""
    phase_kz_col([::Type{TA},] sz::NTuple{N, Int}; Δz=one(eltype(arr)), ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}

Calculates a propagation phase (without the 2pi factor!) for a given z-position, which can be defined via Δz supplied to the function.
By default, Nyquist sampling it is assumed such that the lateral k_xy corresponds to the XY border in frequency space at the edge 
of the Ewald circle.
However, via the xy `scale` entries the k_max can be set appropriately. The propagation equation should
Δz .* sqrt.(1-kxy_rel^2) as the propagation phase. The Z-propagation distance (Δz) has to be specified in 
units of the wavelength in the medium (`λ = n*λ₀`).
Note that since the phase is normalized to 1 instead of 2pi, you need to use this phase in the following sense: `cispi.(2.*phase_kz(...))`.

If the array has 3 dimensions, a stack of equal-distance propagation phases will be generated with the slice
sz[3]÷2+1  corresponding to the mid position yielding no phase change.

#Arguments
+ `TA`:     Array type of the result array. For cuda calculations use `CuArray{Float32}`.
+ `sz`:     Size (2D) of the result array. 
+ `Δz`:     distance in Z to propagate per slice in relation to the wavelength. Nyquist sampling would be 0.5.
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate. By default, the (Fourier space) center position along Z is chosen.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.
"""
function phase_kz_col(::Type{TA}, sz::NTuple{N, Int}; Δz=one(eltype(arr)), ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1)), use_sep=false) where{TA, N}
    if length(sz) > 3
        error("phase_kz are only allowed up to the third dimension. If you need to propagate several stacks, use broadcasting.")
    end
    arr = TA(undef, sz)
    phase_kz_col!(arr; Δz=Δz, ref_idx=ref_idx, k_max=k_max, scale=scale, use_sep=use_sep) 
end
    
"""
    phase_kz_col!(arr::AbstractArray{T,N}; Δz=one(eltype(arr)), ref_idx = size(arr,3)÷2+1, k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}

Calculates a propagation phase (without the 2pi factor!) for a given z-position, which can be defined via Δz supplied to the function.
By default, Nyquist sampling it is assumed such that the lateral k_xy corresponds to the XY border in frequency space at the edge 
of the Ewald circle.
However, via the xy `scale` entries the k_max can be set appropriately. The propagation equation uses
Δz .* sqrt.(1-kxy_rel^2) as the propagation phase. The Z-propagation distance (Δz) has to be specified in 
units of the wavelength in the medium (`λ = n*λ₀`).
Note that since the phase is normalized to 1 instead of 2pi, you need to use this phase in the following sense: `cispi.(2.*phase_kz(...))`.

If `arr` has 3 dimensions, a stack of equal-distance propagation phases will be generated with the slice
size(arr,3)÷2+1  corresponding to the mid position yielding no phase change.

#Arguments
+ `arr`:    the array to fill with propagators. If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `Δz`:     distance in Z to propagate per slice in relation to the wavelength. Nyquist sampling would be 0.5.
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate (i.e. the phase is zero). By default, the (Fourier space) center position along Z is chosen.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.
"""
function phase_kz_col!(arr::AbstractArray{T,N}; Δz=one(eltype(arr)), ref_idx = size(arr,3)÷2+1, k_max=0.5f0, scale=T.(0.5 ./ max.(size(arr) ./ 2, 1)), use_sep=false) where{T, N}
    # function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1))) where{TA, N}
    # if any(offset[1:2] .!= size(arr)[1:2].÷2 .+1)
    #     error("offset[1:2] needs to be size(arr)[1:2].÷2 .+1 to preserve radial symmetry for phase_kz_col().")
    # end
    RT = real(eltype(arr))
    sz = size(arr)
    if isa(k_max, NTuple{2})
        scale = scale[1:2] .* RT(0.5) ./ k_max[1:2];
        k_max = RT(0.5)
    end
    # Δz= 1 # length(offset) > 2 ? RT(offset[3]) : one(RT)
    k2_max = RT.(k_max .^2)
    if (length(sz) < 3 || sz[3] == 1)
        fac = RT(2Δz)
        # f(r2) = sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac
        f(r2) = (r2 >= k2_max) ? zero(real(eltype(arr))) : fac * sqrt(k2_max - r2)
        if (use_sep)
            rr2sep = rr2_sep(real_arr_type(typeof(arr)), sz[1:2]; scale = scale) 
            arr .= f.(rr2sep); 
            return arr
        else
            return calc_radial2_symm!(arr, f; scale=scale); 
        end
    else
        zref = (ref_idx < sz[3]) ? ref_idx + 1 : ref_idx - 1; # The plane which contains one single propagation operation
        fac = (ref_idx < sz[3]) ? RT(2Δz) : RT(-2Δz);
        # g(r2) = sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac
        g(r2) = (r2 >= k2_max) ? zero(real(eltype(arr))) : fac * sqrt(k2_max - r2)
        ref_slice = @view arr[:,:,zref];
        if (use_sep)
            rr2sep = rr2_sep(real_arr_type(typeof(arr)), sz[1:2]; scale = scale) 
            ref_slice .= g.(rr2sep); 
        else
            calc_radial2_symm!(ref_slice, g; scale=scale); 
        end
        zdist = (ref_idx < sz[3]) ? reorient((1:sz[3]) .- ref_idx,Val(3)) : reorient(.-((1:sz[3]) .- ref_idx), Val(3)) 
        # arr .= zdist .* ref_slice
        arr[:,:,1:zref-1] .= zdist[:,:,1:zref-1] .* ref_slice
        arr[:,:,zref+1:end] .= zdist[:,:,zref+1:end] .* ref_slice
        return arr
    end
end

"""
    phase_kz_col!(arr::AbstractArray{T,N}, sampling::NTuple{3}, λ; ref_idx = size(arr,3)÷2+1, use_sep=false) where{T, N}

Calculates a propagation phase (without the 2pi factor!) for a given z-position, which can be defined via `λ` and `sampling`.

Note that since the phase is normalized to 1 instead of 2pi, you need to use this phase in the following sense: `cispi.(2.*phase_kz(...))`.

If `arr` has 3 dimensions, a stack of equal-distance propagation phases will be generated with the slice
size(arr,3)÷2+1  corresponding to the mid position yielding no phase change.

#Arguments
+ `arr`:    the array to fill with propagators. If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `sampling`: pixelsize (e.g. in micrometer)
+ `λ`:  wavelength (same units, i.e. micrometer).
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate (i.e. the phase is zero). By default, the (Fourier space) center position along Z is chosen.
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.
"""
function phase_kz_col!(arr::AbstractArray{T,N}, sampling::NTuple{3}, λ; ref_idx = size(arr,3)÷2+1, use_sep=false) where{T, N}
    return phase_kz_col!(arr; Δz = sampling[3] ./ λ, k_max = sampling[1:2] ./ λ, ref_idx=ref_idx, use_sep=use_sep)
end

"""
    phase_kz_col(sz::NTuple{N, Int}; Δz=one(eltype(DefaultArrType)), ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, k_max=0.5f0, scale=0.5 ./ (max.(sz ./ 2, 1)), use_sep=false) where{N}

Calculates a propagation phase (without the 2pi factor!) for a given z-position, which can be defined via Δz supplied to the function.
By default, Nyquist sampling it is assumed such that the lateral k_xy corresponds to the XY border in frequency space at the edge 
of the Ewald circle.
However, via the xy `scale` entries the k_max can be set appropriately. The propagation equation uses
Δz .* sqrt.(1-kxy_rel^2) as the propagation phase. The Z-propagation distance (Δz) has to be specified in 
units of the wavelength in the medium (`λ = n*λ₀`).
Note that since the phase is normalized to 1 instead of 2pi, you need to use this phase in the following sense: `cispi.(2.*phase_kz(...))`.

#Arguments
+ `sz`:     Size (2D) of the result array. 
+ `Δz`:     distance in Z to propagate per slice in relation to the wavelength. Nyquist sampling would be 0.5.
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate (i.e. the phase is zero). By default, the (Fourier space) center position along Z is chosen.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.
"""
function phase_kz_col(sz::NTuple{N, Int}; Δz=one(eltype(DefaultArrType)), ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, k_max=0.5f0, scale=0.5 ./ (max.(sz ./ 2, 1)), use_sep=false) where{N}
    return phase_kz_col(DefaultArrType, sz; Δz=Δz, ref_idx = ref_idx, k_max=k_max, scale=scale, use_sep=use_sep)
end

"""
    phase_kz_col([::Type{TA},] sz::NTuple{N, Int}, sampling::NTuple{3}, λ; ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=false) where{TA, N}

Calculates a propagation phase (without the 2pi factor!) for a given z-position, which can be defined via `λ` and `sampling`.

Note that since the phase is normalized to 1 instead of 2pi, you need to use this phase in the following sense: `cispi.(2.*phase_kz(...))`.

If `arr` has 3 dimensions, a stack of equal-distance propagation phases will be generated with the slice
size(arr,3)÷2+1  corresponding to the mid position yielding no phase change.

#Arguments
+ `TA`:     Array type of the result array. For cuda calculations use `CuArray{Float32}`. By default Float32 is used.
+ `sz`:     Size (2D) of the result array. 
+ `sampling`: pixelsize (e.g. in micrometer)
+ `λ`:  wavelength (same units, i.e. micrometer).
+ `ref_idx`: reference index at which the propagator has no effect. E.g. `ref_idx=1` means the first slice of the result array does not propagate (i.e. the phase is zero). By default, the (Fourier space) center position along Z is chosen.
+ `use_sep`: This boolean flag switches to an algorithm using rr2_sep and no corner copies. In CUDA this is a little faster.
"""
function phase_kz_col(::Type{TA}, sz::NTuple{N, Int}, sampling::NTuple{3}, λ; ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=false) where{TA, N}
    if length(sz) > 3
        error("propagators are only allowed up to the third dimension. If you need to propagate several stacks, use broadcasting.")
    end
    arr = TA(undef, sz)
    phase_kz_col!(arr, sampling, λ; ref_idx=ref_idx, use_sep=use_sep)
end

function phase_kz_col(sz::NTuple{N, Int}, sampling::NTuple{3}, λ; ref_idx = (length(sz) < 3) ? 1 : sz[3]÷2+1, use_sep=false) where{N}
    phase_kz_col(DefaultArrType, sz, sampling, λ; ref_idx = ref_idx, use_sep=use_sep)
end
