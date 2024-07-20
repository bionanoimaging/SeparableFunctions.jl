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
        (:(gaussian),(sigma=1.0,), :((x,sz, sigma) -> exp.(.-x.^2 ./(2*sigma^2))), Float32, *, 
            :((f, x, sz, sigma) -> .-x./sigma^2 .* f),
            :((f, x, sz, sigma) -> x.^2 ./sigma^3 .* f)
            ), 
        (:(normal), (sigma=1.0,), :((x,sz, sigma) -> exp.(.- x.^2 ./(2*sigma^2)) ./ (sqrt(eltype(x)(2pi))*sigma)), Float32, *,
            :((f, x, sz, sigma) -> .-x./sigma^2 .* f),
            :((f, x, sz, sigma) -> (x.^2 ./sigma^3 .+ 1/sigma) .* f)
            ),
        (:(sinc), NamedTuple(), :((x,sz) -> sinc.(x)), Float32, *,
            :((f, x, sz) -> ifelse.(x .== zero(eltype(x)), zeros(eltype(x), size(x)), (cospi.(x) .- f)./x))
            ),
        # the value "nothing" means that this default argument will not be handed over. But this works only for the last argument!
        (:(exp_ikx), (shift_by=nothing,), :((x,sz, shift_by=sz÷2) -> cis.(x.*(-eltype(x)(2pi)*shift_by/sz))), ComplexF32, *,
            :((f, x, sz, shift_by) -> (-eltype(x)(2pi)*shift_by/sz) .* f),
            :((f, x, sz, shift_by) -> (-eltype(x)(2pi)/sz) .*x .* f)
            ),
        (:(ramp), (slope=0,), :((x,sz, slope) -> slope.*x), Float32, +,
            :((f, x, sz, slope) ->  slope),
            :((f, x, sz, slope) ->  x)
            ), # different meaning than IFA ramp
        (:(rr2), NamedTuple(), :((x, sz) -> (x.*x)), Float32, +,
            :((f, x, sz) ->  2 .* x)
            ),
        (:(box), (boxsize=nothing,), :((x, sz, boxsize=sz/2) -> abs.(x) .<= (boxsize/2)), Bool, *,
            :((f, x, sz) ->  one(eltype(x)))
            ),
    ]
    return functions
end

# would be nice to have a macro which defines all those function extensions.
# But its not quite that simple. First try:
# macro define_separable(basename, fct, basetype, operation)
#     @eval function $(Symbol(basename, :_col))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
#         fct = $(fct) # to assign the function to a symbol
#         separable_create(TA, fct, sz, args...; operation=$(operation), kwargs...)
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
        # @show "creating rrule for $(Symbol(F[1], :_raw)) "
        @eval function ChainRulesCore.rrule(::typeof($(Symbol(F[1], :_raw))), x, sz; kwargs...) 
            # println("in rrule raw")
            y = $(Symbol(F[1], :_raw))(x, sz; kwargs...) # to assign the function to a symbol
            function mypullback(dy)
                mydx =  dy .* $(F[6])(y, x, sz; kwargs...)
                return NoTangent(), mydx, NoTangent()
            end
            return y, mypullback
        end
        # @show "added rrule for $(Symbol(F[1], :_raw))"
    end
    if (length(F) == 7) # a gradient definition was provided explicitely
        # @show "creating rrule for $(Symbol(F[1], :_raw)) "
        @eval function ChainRulesCore.rrule(::typeof($(Symbol(F[1], :_raw))), x, sz, args...; kwargs...) 
            # println("in rrule raw")
            y = $(Symbol(F[1], :_raw))(x, sz, args...; kwargs...) # to assign the function to a symbol
            function mypullback(dy)
                # println("pb")
                # @show dy
                # @show $(F[6])(y, x, sz, args...; kwargs...)
                mydx =  dy .* $(F[6])(y, x, sz, args...; kwargs...)
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

    @eval function $(Symbol(F[1], :_col))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_create(TA, fct, sz, args...; defaults=$(F[2]), operation=$(F[5]), kwargs...)
    end
 
    @eval function $(Symbol(F[1], :_col))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_create(Array{$(F[4])}, fct, sz, args...; defaults=$(F[2]), operation=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_sep))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        calculate_broadcasted(TA, fct, sz, args...; defaults=$(F[2]), operation=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_sep))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        calculate_broadcasted(Array{$(F[4])}, fct, sz, args...; defaults=$(F[2]), operation=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_nokw_sep))(::Type{TA}, sz::NTuple{N, Int}, args...;
                        all_axes = (similar_arr_type(TA, eltype(TA), Val(1)))(undef, sum(sz))
                    ) where {TA, N}
        # fct = $(F[3]) # to assign the function to a symbol

        # @show "call"
        return calculate_broadcasted_nokw(TA, $(Symbol(F[1], :_raw)), sz, args...; defaults=$(F[2]), operation=$(F[5]), all_axes=all_axes)
        # operation=$(F[5])
        # return calculate_separables_nokw(TA, fct, sz, args...; all_axes=all_axes), operation
    end

    @eval function $(Symbol(F[1], :_nokw_sep))(sz::NTuple{N, Int}, args...;
                        all_axes = (similar_arr_type(Array{$(F[4])}, eltype(Array{$(F[4])}), Val(1)))(undef, sum(sz))
                    ) where {N}
        # fct = $(F[3]) # to assign the function to a symbol        
        # @show "call2"
        return calculate_broadcasted_nokw(Array{$(F[4])}, $(Symbol(F[1], :_raw)), sz, args...; defaults=$(F[2]), operation=$(F[5]), all_axes=all_axes)
        # operation=$(F[5])
        # return calculate_separables_nokw(Array{$(F[4])}, fct, sz, args...; all_axes=all_axes), operation
    end
 
    @eval function $(Symbol(F[1], :_lz))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_view(TA, fct, sz, args...; defaults=$(F[2]), operation=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_lz))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_view(Array{$(F[4])}, fct, sz, args...; defaults=$(F[2]), operation=$(F[5]), kwargs...)
    end

    # collected: fast separable calculation but resulting in an ND array
    @eval export $(Symbol(F[1], :_col))
    # separated: a vector of separated contributions is returned and the user has to combine them
    @eval export $(Symbol(F[1], :_sep))
    # lazy: A LazyArray representation is returned
    @eval export $(Symbol(F[1], :_nokw_sep))
    # @eval export $(Symbol(F[1], :_lz))
end 


## Here some individual versions based on copy_corners! stuff. They only exist in the _cor version as they are not separable in X and Y.
"""
    propagator_col([]::Type{TA},] sz::NTuple{N, Int}; Δz=one(eltype(TA)), k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}

generates a propagator for propagating optical fields via exp(i kz Δz) with kz=sqrt(k0^2-kx^2-ky^2). The k-space radius is stated by
k_max relative to the Nyquist frequency, as long as the scale remains to be 1 ./ (2 max.(sz ./ 2, 1))).

#Arguments
+ `TA`:     type of the array to generate. E.g. Array{Float64} or CuArray{Float32}.
+ `sz`:     size of the array to generate.  If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `Δz`:     distance in Z to propagate per slice.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
"""
function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=one(eltype(TA)), k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}
# function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1))) where{TA, N}
    if length(sz) > 3
        error("propagators are only allowed up to the third dimension. If you need to propagate several stacks, use broadcasting.")
    end
    arr = TA(undef, sz)
    propagator_col!(arr; Δz=Δz, k_max=k_max, scale=scale) 
end

function propagator_col(sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1))) where{N}
    propagator_col(DefaultComplexArrType, sz; Δz=Δz, k_max=k_max, scale=scale)
end

"""
    propagator_col!(arr::AbstractArray{T,N}; Δz=one(eltype(TA)), k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}

generates a propagator for propagating optical fields via exp(i kz Δz) with kz=sqrt(k0^2-kx^2-ky^2). The k-space radius is stated by
k_max relative to the Nyquist frequency, as long as the scale remains to be 1 ./ (2 max.(sz ./ 2, 1))).

#Arguments
+ `arr`:    the array to fill with propagators. If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `Δz`:     distance in Z to propagate per slice.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
"""    
function propagator_col!(arr::AbstractArray{T,N}; Δz=one(eltype(arr)), k_max=0.5f0, scale=0.5f0 ./ (max.(size(arr) ./ 2, 1))) where{T, N}
    # function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1))) where{TA, N}
    k2_max = real(eltype(arr))(k_max .^2)
    # fac = eltype(arr)(4im * pi * Δz)
    # f(r2) = cispi(sqrt(max(zero(real(eltype(TA))),k2_max - r2)) * (4 * Δz))
    # f(r2) = exp(sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac)
    fac = real(eltype(arr))(4pi * Δz)
    f(r2) = cis(sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac)
    if length(size(arr)) < 3 || sz[3] == 1
        return calc_radial2_symm!(arr, f; scale=scale); 
    else
        zmid = size(sz,3)÷2+1
        calc_radial2_symm!((@view arr[:,:,zmid+1]), f; scale=scale); 
        for z=1:size(sz,3)
            if z != zmid+1
                arr[:,:,z] .= (z-zmid) .* (@view arr[:,:,zmid+1])
            end
        end
    end
end
    
"""
    phase_kz_col([::Type{TA},] sz::NTuple{N, Int};k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}

Calculates a propagation phase (without the 2pi factor!) for a given z-position, which can be defined via a 3rd entry in the `offset` supplied to the function.
By default, Nyquist sampling it is assumed such that the lateral k_xy corresponds to the XY border in frequency space at the edge 
of the Ewald circle.
However, via the xy `scale` entries the k_max can be set appropriately. The propagation equation should
Δz .* sqrt.(1-kxy_rel^2) as the propagation phase. The Z-propagation distance (Δz) has to be specified in 
units of the wavelength in the medium (`λ = n*λ₀`).
Note that since the phase is normalized to 1 instead of 2pi, you need to use this phase in the following sense: `cispi.(2.*phase_kz(...))`.

#Arguments
+ `TA`:     Array type of the result array. For cuda calculations use `CuArray{Float32}`.
+ `sz`:     Size (2D) of the result array. 
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
"""
function phase_kz_col(::Type{TA}, sz::NTuple{N, Int}; k_max=1f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}
    if length(sz) > 3
        error("phase_kz are only allowed up to the third dimension. If you need to propagate several stacks, use broadcasting.")
    end
    arr = TA(undef, sz)
    phase_kz_col!(arr; k_max=k_max, scale=scale) 
end
    
"""
    phase_kz_col!(arr::AbstractArray{T,N}; k_max=0.5f0, scale=0.5f0 ./ (max.(sz ./ 2, 1))) where{TA, N}

Calculates a propagation phase (without the 2pi factor!) for a given z-position, which can be defined via a 3rd entry in the `offset` supplied to the function.
By default, Nyquist sampling it is assumed such that the lateral k_xy corresponds to the XY border in frequency space at the edge 
of the Ewald circle.
However, via the xy `scale` entries the k_max can be set appropriately. The propagation equation uses
Δz .* sqrt.(1-kxy_rel^2) as the propagation phase. The Z-propagation distance (Δz) has to be specified in 
units of the wavelength in the medium (`λ = n*λ₀`).
Note that since the phase is normalized to 1 instead of 2pi, you need to use this phase in the following sense: `cispi.(2.*phase_kz(...))`.

#Arguments
+ `arr`:    the array to fill with propagators. If a 3rd dimension is present, a stack a propagators is returned, one for each multiple of Δz.
+ `k_max`:  maximum propagation radius in k-space. I.e. limit of the k-sphere. This is not the aperture limit!
+ `scale`:  specifies how to interpret k-space positions. Should remain to be 1 ./ (2 max.(sz ./ 2, 1))).
"""
function phase_kz_col!(arr::AbstractArray{T,N}; k_max=1.0f0, scale=T(0.5 ./ (max.(size(arr) ./ 2), 1))) where{T, N}
    # function propagator_col(::Type{TA}, sz::NTuple{N, Int}; Δz=1.0, k_max=0.5, scale=0.5 ./ (max.(sz ./ 2, 1))) where{TA, N}
    # if any(offset[1:2] .!= size(arr)[1:2].÷2 .+1)
    #     error("offset[1:2] needs to be size(arr)[1:2].÷2 .+1 to preserve radial symmetry for phase_kz_col().")
    # end
    RT = real(eltype(arr))
    # Δz= 1 # length(offset) > 2 ? RT(offset[3]) : one(RT)
    k2_max = RT(k_max .^2)
    # fac = eltype(arr)(4im * pi * Δz)
    # f(r2) = cispi(sqrt(max(zero(real(eltype(TA))),k2_max - r2)) * (4 * Δz))
    # f(r2) = exp(sqrt(max(zero(real(eltype(arr))),k2_max - r2)) * fac)
    # fac = RT(Δz)
    f(r2) = sqrt(max(zero(real(eltype(arr))),k2_max - r2)) # * fac
    if length(size(arr)) < 3 || sz[3] == 1
        return calc_radial2_symm!(arr, f; scale=scale); 
    else
        zmid = size(sz,3)÷2+1
        calc_radial2_symm!((@view arr[:,:,zmid+1]), f; scale=scale); 
        for z=1:size(sz,3)
            if z != zmid+1
                arr[:,:,z] .= (z-zmid) .* (@view arr[:,:,zmid+1])
            end
        end
    end
end

function phase_kz_col(sz::NTuple{N, Int}; k_max=1.0f0, scale=0.5 ./ (max.(sz ./ 2, 1))) where{N}
    phase_kz_col(DefaultArrType, sz; k_max=k_max, scale=scale)
end
