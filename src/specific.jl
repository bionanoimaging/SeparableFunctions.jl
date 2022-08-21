# function gaussian_sep(::Type{TA}, sz::NTuple{N, Int}, sigma, pos=zeros(Float32,N); offset=sz.÷2 .+1) where {TA, N}
#     invsigma22 = 1 ./(2 .*sigma.^2)
#     fct = (r, invsigma22, pos) -> exp(-(r-pos)^2*invsigma22)
#     separable_create(TA, fct, sz, invsigma22, pos; offset=offset)
# end

# function gaussian_sep(sz::NTuple{N, Int}, sigma, pos=zeros(Float32,N); offset=sz.÷2 .+1) where {N}
#     gaussian_sep(DefaultArrType, sz, sigma, pos; offset=offset)
# end

# function gaussian_sep_lz(::Type{TA}, sz::NTuple{N, Int}, sigma, pos=zeros(Float32,N); offset=sz.÷2 .+1) where {TA, N}
#     invsigma22 = 1 ./(2 .*sigma.^2)
#     fct = (r, invsigma22, pos) -> exp(-(r-pos)^2*invsigma22)
#     separable_view(TA, fct, sz, invsigma22, pos; offset=offset)
# end

# function gaussian_sep_lz(sz::NTuple{N, Int}, sigma, pos=zeros(Float32,N); offset=sz.÷2 .+1) where {N}
#     gaussian_sep_lz(DefaultArrType, sz, sigma, pos; offset=offset)
# end

function generate_functions_expr()
    # offset and scale is already wrapped in the generator function
    # x_expr = :(scale .* (x .- offset))

    functions = [
        # function_name, function_expression (always startin with x, sz, ...), result_type:
        # Rules: the calculation function has no kwargs but the last N arguments are the kwargs of the wrapper function
        (:(gaussian), :((x,sz;sigma=1.0) -> exp(- x^2/(2 .* sigma^2))), Float32, *),
        (:(normal), :((x,sz;sigma=1.0) -> exp(- x^2/(2 .* sigma^2)) / (sqrt(typeof(x)(2pi))*sigma)), Float32, *),
        (:(sinc), :((x,sz) -> sinc(x)), Float32, *),
        (:(exp_ikx), :((x,sz; shift_by=sz÷2) -> cis(x*(-2pi*shift_by/sz))), ComplexF32, *),
        (:(ramp), :((x,sz; slope) -> slope*x), Float32, +), # different meaning than IFA ramp
        (:(rr2), :((x,sz) -> (x*x)), Float32, +),
        (:(box), :((x,sz; boxsize=sz/2) -> abs(x) <= (boxsize/2)), Bool, *),
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
 
    @eval function $(Symbol(F[1], :_col))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[2]) # to assign the function to a symbol
        separable_create(TA, fct, sz, args...; operation=$(F[4]), kwargs...)
    end
 
    @eval function $(Symbol(F[1], :_col))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[2]) # to assign the function to a symbol
        separable_create(Array{$(F[3])}, fct, sz, args...; operation=$(F[4]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_sep))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[2]) # to assign the function to a symbol
        calculate_separables(TA, fct, sz, args...; kwargs...)
    end

    @eval function $(Symbol(F[1], :_sep))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[2]) # to assign the function to a symbol
        calculate_separables(Array{$(F[3])}, fct, sz, args...; kwargs...)
    end
 
    @eval function $(Symbol(F[1], :_lz))(::Type{TA}, sz::NTuple{N, Int}, args...; kwargs...) where {TA, N}
        fct = $(F[2]) # to assign the function to a symbol
        separable_view(TA, fct, sz, args...; operation=$(F[4]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_lz))(sz::NTuple{N, Int}, args...; kwargs...) where {N}
        fct = $(F[2]) # to assign the function to a symbol
        separable_view(Array{$(F[3])}, fct, sz, args...; operation=$(F[4]), kwargs...)
    end

    # collected: fast separable calculation but resulting in an ND array
    @eval export $(Symbol(F[1], :_col))
    # separated: a vector of separated contributions is returned and the user has to combine them
    @eval export $(Symbol(F[1], :_sep))
    # lazy: A LazyArray representation is returned
    @eval export $(Symbol(F[1], :_lz))
end 
