common_docstring = "To create the array on the GPU
supply for example CuArray{Float32} as the array type. Note that the `_col` version yields a collected array, the `_lz` version a `LazyArray` and
the `_sep` version an iterable of one-dimensional but oriented arrays, which can be used via the `.*(res...)` syntax with `res` being the result of the `_sep` call.

# Arguments
+ `TA`:     optionally an array type can be supplied. The default is Array{Float32}
+ `sz`:     size of the result array

+ `pos`:    position of the gaussian in relationship to the center defined by `offset`
+ `offset`: center position of the array
"

returns_col = "returns the collected N-dimensional array."
returns_lz = "returns the a lazy version of an N-dimensional array only using memory for the separated 1-dimensional arrays. See the `_col` version for an example."
returns_sep_mul = "returns an iterable with the one-dimensional arrays, which can be used via `.*(res...)` wiht `res` being the result of calling this function."
returns_sep_add = "returns an iterable with the one-dimensional arrays, which can be used via `.+(res...)` wiht `res` being the result of calling this function."


gaussian_docstring = "creates a multidimensional Gaussian by exployting separability speeding up the calculation."

"""
    gaussian_col([::Type{TA},] sz::NTuple{N, Int}; sigma=ones(eltype(TA),N), pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(gaussian_docstring)$(common_docstring)

+ `sigma`:  tuple of standard-deviation along each dimensional
$(returns_col)

#Example
```jldoctest
julia> pos = (1.1, 0.2); sigma = (0.5, 1.0);
julia> my_gaussian = gaussian_col((6,5); sigma=sigma, pos=pos)
6×5 Matrix{Float32}:
    2.22857f-16  1.21991f-15  2.4566f-15   1.81989f-15  4.95978f-16
    3.99823f-10  2.18861f-9   4.40732f-9   3.26502f-9   8.89822f-10
    1.3138f-5    7.19168f-5   0.000144823  0.000107287  2.92392f-5
    0.00790705   0.0432828    0.0871608    0.0645703    0.0175975
    0.0871608    0.477114     0.960789     0.71177      0.19398
    0.0175975    0.0963276    0.19398      0.143704     0.0391639
```
"""
gaussian_col

"""
gaussian_lz([::Type{TA},] sz::NTuple{N, Int}; sigma=ones(eltype(TA),N), pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(gaussian_docstring)$(common_docstring)

+ `sigma`:  tuple of standard-deviation along each dimensional
$(returns_lz)
"""
gaussian_lz

"""
gaussian_sep([::Type{TA},] sz::NTuple{N, Int}; sigma=ones(eltype(TA),N), pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(gaussian_docstring)$(common_docstring)

+ `sigma`:  tuple of standard-deviation along each dimensional
$(returns_sep_mul)
"""
gaussian_sep
###
normal_docstring = "creates a multidimensional normalized Gaussian by exployting separability speeding up the calculation. The integral of the multidimensional array, if it where infinite, is normalized to one."

"""
    normal_col([::Type{TA},] sz::NTuple{N, Int}; sigma=ones(eltype(TA),N), pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(normal_docstring)$(common_docstring)

+ `sigma`:  tuple of standard-deviation along each dimensional
$(returns_col)
#Example
```jldoctest
julia> pos = (1.1, 0.2); sigma = (0.5, 1.0);
julia> my_normal = normal_col((6,5); sigma=sigma, pos=pos)
6×5 Matrix{Float32}:
    7.09377f-17  3.88309f-16  7.81959f-16  5.79289f-16  1.57875f-16
    1.27268f-10  6.96656f-10  1.40289f-9   1.03929f-9   2.83239f-10
    4.18196f-6   2.28918f-5   4.60985f-5   3.41506f-5   9.30713f-6
    0.00251689   0.0137773    0.0277442    0.0205534    0.00560145
    0.0277442    0.15187      0.305829     0.226564     0.0617458
    0.00560145   0.030662     0.0617458    0.0457424    0.0124663
```
"""
normal_col

"""
    normal_lz([::Type{TA},] sz::NTuple{N, Int}; sigma=ones(eltype(TA),N), pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(normal_docstring)$(common_docstring)

+ `sigma`:  tuple of standard-deviation along each dimensional
$(returns_lz)
"""
normal_lz

"""
normal_sep([::Type{TA},] sz::NTuple{N, Int}; sigma=ones(eltype(TA),N), pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(normal_docstring)$(common_docstring)

+ `sigma`:  tuple of standard-deviation along each dimensional
$(returns_sep_mul)
"""
normal_sep

###
rr2_docstring = "yields the absolute square of the distance to the zero-position."

"""
    rr2_col([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(rr2_docstring)$(common_docstring)
$(returns_col)
#Example
```jldoctest
julia> pos = (1.1, 0.2); 
julia> my_rr2 = rr2_col((6,5); pos=pos)
6×5 Matrix{Float32}:
    21.65  18.25  16.85  17.45  20.05
    14.45  11.05   9.65  10.25  12.85
    9.25   5.85   4.45   5.05   7.65
    6.05   2.65   1.25   1.85   4.45
    4.85   1.45   0.05   0.65   3.25
    5.65   2.25   0.85   1.45   4.05
```
"""
rr2_col

"""
    rr2_lz([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(rr2_docstring)$(common_docstring)
$(returns_lz)
"""
rr2_lz

"""
    rr2_sep([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(rr2_docstring)$(common_docstring)
$(returns_sep_add)
"""
rr2_sep

###
sinc_docstring = "yields the outer product of sinc functions. This corresponds to the diffraction pattern of a rectangular aperture. Note that it is not circularly symmetric."

"""
    sinc_col([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(sinc_docstring)$(common_docstring)
$(returns_col)
#Example
```jldoctest
julia> pos = (1.1, 0.2); 
julia> my_sinc = sinc_col((6,5); pos=pos)
6×5 Matrix{Float32}:
  0.0020403   -0.00374056   0.0224433   0.00561083  -0.0024937
 -0.00269847   0.00494719  -0.0296831  -0.00742078   0.00329812
  0.00398345  -0.00730299   0.0438179   0.0109545   -0.00486866
 -0.00760477   0.0139421   -0.0836524  -0.0209131    0.00929472
  0.0836524   -0.153363     0.920177    0.230044    -0.102242
  0.00929472  -0.0170403    0.102242    0.0255605   -0.0113602
```
"""
sinc_col

"""
sinc_lz([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(sinc_docstring)$(common_docstring)
$(returns_lz)
"""
sinc_lz

"""
sinc_sep([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(sinc_docstring)$(common_docstring)
$(returns_sep_mul)
"""
sinc_sep

###
box_docstring = "creates a Boolean box, being True inside and False outside. The side-length of the box can be defined via the argument `boxsize`."

"""
    box_col([::Type{TA},] sz::NTuple{N, Int}; boxsize=sz./2, pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(box_docstring)$(common_docstring)

+ `boxsize`:  a vector defining each sidelength of the box.
$(returns_col)
#Example
```jldoctest
julia> pos = (1.1, 0.2); 
julia> my_box = box_col((6,5); pos=pos)
6×5 BitMatrix:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  1  1  1  0
 0  1  1  1  0
 0  1  1  1  0
```
"""
box_col

"""
box_lz([::Type{TA},] sz::NTuple{N, Int}; boxsize=sz./2, pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(box_docstring)$(common_docstring)

+ `boxsize`:  a vector defining each sidelength of the box.
$(returns_lz)
"""
box_lz

"""
box_sep([::Type{TA},] sz::NTuple{N, Int}; boxsize=sz./2, pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(box_docstring)$(common_docstring)

+ `boxsize`:  a vector defining each sidelength of the box.
$(returns_sep_mul)
"""
box_sep

###
ramp_docstring = "creates an N-dimensional ramp along the gradient-direction defined by `slope`. Note that this disagrees with the nomenclature of the ramp-function defined in `IndexFunArrays.jl`."

"""
ramp_col([::Type{TA},] sz::NTuple{N, Int}; slope, pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(ramp_docstring)$(common_docstring)

+ `slope`:  a vector defining the N-dimensional gradient of the ramp.
$(returns_col)
#Example
julia> my_ramp = ramp_col((6,5); slope=(0.0,0.5), pos=pos)
6×5 Matrix{Float32}:
 -1.1  -0.6  -0.1  0.4  0.9
 -1.1  -0.6  -0.1  0.4  0.9
 -1.1  -0.6  -0.1  0.4  0.9
 -1.1  -0.6  -0.1  0.4  0.9
 -1.1  -0.6  -0.1  0.4  0.9
 -1.1  -0.6  -0.1  0.4  0.9
```
"""
ramp_col

"""
ramp_lz([::Type{TA},] sz::NTuple{N, Int}; slope, pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(ramp_docstring)$(common_docstring)

+ `slope`:  a vector defining the N-dimensional gradient of the ramp.
$(returns_lz)
"""
ramp_lz

"""
ramp_sep([::Type{TA},] sz::NTuple{N, Int}; slope, pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(ramp_docstring)$(common_docstring)

+ `slope`:  a vector defining the N-dimensional gradient of the ramp.
$(returns_sep_mul)
"""
ramp_sep

###
exp_ikx_docstring = "yield an `exp.(1im .* k .* Δx)` function. A scaling of one means that the `shift_by` argument corresponds to `Δx` in integer pixels, but the term is in Fourier space."

"""
    exp_ikx_col([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(exp_ikx_docstring)$(common_docstring)

+ `shift_by`:  a vector defining the real-space shift that would be caused by this function being multiplied in Fourier-space.
$(returns_col)
#Example
```jldoctest
julia> pos = (1.1, 0.2); 
julia> my_exp_ikx = exp_ikx_col((6,5); shift_by=(1.0,0.0), pos=pos)
6×5 Matrix{ComplexF32}:
 -0.406737-0.913545im  -0.406737-0.913545im  -0.406737-0.913545im  -0.406737-0.913545im  -0.406737-0.913545im
 -0.994522-0.104528im  -0.994522-0.104528im  -0.994522-0.104528im  -0.994522-0.104528im  -0.994522-0.104528im
 -0.587785+0.809017im  -0.587785+0.809017im  -0.587785+0.809017im  -0.587785+0.809017im  -0.587785+0.809017im
  0.406737+0.913545im   0.406737+0.913545im   0.406737+0.913545im   0.406737+0.913545im   0.406737+0.913545im
  0.994522+0.104528im   0.994522+0.104528im   0.994522+0.104528im   0.994522+0.104528im   0.994522+0.104528im
  0.587785-0.809017im   0.587785-0.809017im   0.587785-0.809017im   0.587785-0.809017im   0.587785-0.809017im
```
"""
exp_ikx_col

"""
    exp_ikx_lz([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(box_docstring)$(common_docstring)

+ `shift_by`:  a vector defining the real-space shift that would be caused by this function being multiplied in Fourier-space.
$(returns_lz)
"""
exp_ikx_lz

"""
    exp_ikx_sep([::Type{TA},] sz::NTuple{N, Int}; pos=zeros(eltype(TA),N), offset=sz.÷2 .+1, scale=1.0) where {TA, N}

$(exp_ikx_docstring)$(common_docstring)

+ `shift_by`:  a vector defining the real-space shift that would be caused by this function being multiplied in Fourier-space.
$(returns_sep_mul)
"""
exp_ikx_sep
