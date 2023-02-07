"""
Calculates multidimensional functions faster by exploiting their separability.
Often a function involves an operation such as a (complex) exponential which by itself is computationally relatively heavy. Yet a number of multidimenstional functions are separable, which means that they can be written as a product (or sum) of single-dimensional functions. A good example of a separable function is a Gaussian function, which can be written as a product of a purely X-dependend Gaussian with a purely Y-dependend Gaussian.

In this package, multidimensional functions are computed by first calculating their single-dimensional values and then creating the final multidimensional result by an outer product. Since multiplications and the broadcasting mechanism of Julia are fast compared to the evaluation of the function at each multidimensional position, the final result is calculated faster. The typical speedup can be an order of magnitude.

The package offers a general way of calculating separable functions as well as a `LazyArrays` version of that function which can then be used inside other expressions.
The non-lazy version should currently also work with `CUDA.jl`, however the `LazyArrays` version does not. To nevertheless use separable expressions in `CUDA.jl`, you can reside to externally applying the broadcast operator to the separable expression (see the `gaussian_sep` example below).

The package further offers a number of predifined separable function implementations such as `gaussian_col()` collects a multidimensional array of a multidimensional Gaussian via a fast seperable implementation, `gaussian_lz()` yields a lazy representation via `LazyArrays` and `sep = gaussian_sep()` yields an iterable of separable pre-oriented vectors which can easily be mutually combined via `.*(sep...)`.

Another noteworthy example is the complex plain wave as represented by the respective function `exp_ikx_col()`, `exp_ikx_lz()`, `exp_ikx_sep()`.

All separable functions share a common interface with
Standard non-named arguments
+ the first optional argument being the type of the result array. Examples are `Array{Float64}`, `CuArray{Float32}` and the default depends on the function but uses a 32-bit result type where applicable.
+ the next argument is the `size` of the result data supplied as a tuple.
+ optionally a further argument can specify the `position` of zero of the array with respect to `offset` as given below. This allows for convenient N-dimensional shifting of the functions. 
Named arguments:
+ `offset`: is a optional named argument specifying center of the array. By default the Fourier center, `size().÷2 .+ 1` is chosen.
+ `scale`: multiplies the axes with these factors. This can be interpreted as a `pixelsize`.
Some functions have additional named arguments. E.g. `gaussian` has the additional named argument `sigma` specifying the width of the gaussian, even though this can also be controlled via `scale`. 
Note that this nomenclature is in large parts identical to the package `IndexFunArrays.jl`.

In general arguments can be supplied as single scalar values or vectors. If a scalar value is supplied, it will automatically be replicated as a vector. E.g. `sigma=10.0` for a  2-dimensional array will be interpreted as `sigma=(10.0, 10.0)`.
"""
module SeparableFunctions
using NDTools, LazyArrays
using ImageTransformations, StaticArrays
using Interpolations

export calculate_separables, separable_view, separable_create
export copy_corners!
export propagator_col, propagator_col!, phase_kz_col, phase_kz_col!

export calc_radial_symm!, calc_radial_symm, get_corner_ranges
export radial_speedup, radial_speedup_ifa
export kwargs_to_args

DefaultResElType = Float32
DefaultArrType = Array{DefaultResElType}
DefaultComplexArrType = Array{complex(DefaultResElType)}

include("utilities.jl")
include("general.jl")
include("specific.jl")
include("radial.jl")
include("exp_iterate.jl")
include("docstrings.jl")

end # module SeparableFunctions
