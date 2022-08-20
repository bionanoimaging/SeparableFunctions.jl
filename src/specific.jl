"""
    gaussian_sep([::Type{TA},] sz::NTuple{N, Int}, sigma, pos=zeros(Float32,N); offset=sz.÷2 .+1, scale=1.0) where {TA, N}

creates a multidimensional Gaussian by exployting separability speeding up the calculation. To create it on the GPU
supply for example CuArray{Float32} as the array type.

# Arguments
+ `TA`:     optionally an array type can be supplied. The default is Array{Float32}
+ `sz`:     size of the result array
+ `sigma`:  tuple of standard-deviation along each dimensional
+ `pos`:    position of the gaussian
```julia
julia> pos = (1.1, 0.2); sigma = (0.5, 1.0);
julia> my_gaussian = gaussian_sep((6,5), sigma, pos)
6×5 Matrix{Float32}:
 2.22857f-16  1.21991f-15  2.4566f-15   1.81989f-15  4.95978f-16
 3.99823f-10  2.18861f-9   4.40732f-9   3.26502f-9   8.89822f-10
 1.3138f-5    7.19168f-5   0.000144823  0.000107287  2.92392f-5
 0.00790705   0.0432828    0.0871608    0.0645703    0.0175975
 0.0871608    0.477114     0.960789     0.71177      0.19398
 0.0175975    0.0963276    0.19398      0.143704     0.0391639```
"""
gaussian_sep
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
        # function_name, extra_variables beyond (x, pos),  function_expression, result_type:
        # Rules: the calculation function has no kwargs but the last N arguments are the kwargs of the wrapper function
        (:(gaussian), (), :((x,pos,sz;sigma=1.0) -> exp(-(x-pos)^2/(2 .* sigma^2))), Float32, *),
        (:(sinc), (:scale,), :((x,pos,sz,scale) -> sinc((x-pos)/scale)), Float32, *),
        (:(exp_ikx), (:Δx,), :((x,pos,sz,Δx) -> cis((pos-x)*(2pi*Δx/sz))), ComplexF32, *),
        (:(ramp), (:slope,), :((x,pos,sz,slope) -> slope*(x-pos)), Float32, +), # different meaning than IFA ramp
        (:(rr2), (), :((x,pos,sz) -> (x*x)), Float32, +),
        (:(box), (:boxsize,), :((x,pos,sz,boxsize) -> abs(x-pos) <= (boxsize/2)), Bool, *),
    ]
    return functions
end

for F in generate_functions_expr() 
    # default functions with offset and scaling behavior
 
    @eval function $(Symbol(F[1], :_col))(::Type{TA}, sz::NTuple{N, Int}, $(F[2]...), pos=zeros(Float32, N); kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_create(TA, fct, sz, pos, sz, $(F[2]...); operation=$(F[5]), kwargs...)
    end
 
    @eval function $(Symbol(F[1], :_col))(sz::NTuple{N, Int}, $(F[2]...), pos=zeros(Float32, N); kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_create(Array{$(F[4])}, fct, sz, pos, sz, $(F[2]...); operation=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_sep))(::Type{TA}, sz::NTuple{N, Int}, $(F[2]...), pos=zeros(Float32, N); kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        calculate_separables(TA, fct, sz, pos, sz, $(F[2]...); kwargs...)
    end
 
    @eval function $(Symbol(F[1], :_sep))(sz::NTuple{N, Int}, $(F[2]...), pos=zeros(Float32, N); kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        calculate_separables(Array{$(F[4])}, fct, sz, pos, sz, $(F[2]...); kwargs...)
    end

    @eval function $(Symbol(F[1], :_lz))(::Type{TA}, sz::NTuple{N, Int}, $(F[2]...), pos=zeros(Float32, N); kwargs...) where {TA, N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_view(TA, fct, sz, pos, sz, $(F[2]...); operation=$(F[5]), kwargs...)
    end

    @eval function $(Symbol(F[1], :_lz))(sz::NTuple{N, Int}, $(F[2]...), pos=zeros(Float32, N); kwargs...) where {N}
        fct = $(F[3]) # to assign the function to a symbol
        separable_view(Array{$(F[4])}, fct, sz, pos, sz, $(F[2]...); operation=$(F[5]), kwargs...)
    end
  
    # collected: fast separable calculation but resulting in an ND array
    @eval export $(Symbol(F[1], :_col))
    # separated: a vector of separated contributions is returned and the user has to combine them
    @eval export $(Symbol(F[1], :_sep))
    # lazy: A LazyArray representation is returned
    @eval export $(Symbol(F[1], :_lz))
end 
